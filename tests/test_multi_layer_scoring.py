"""Scoring several depths from one pass, and filing each under the right one.

Running the model is the whole cost of scoring without a cache, and a pass
already produces every depth, so the depths share one. What that buys is only
worth having if each column ends up under the layer it actually came from: an
off-by-one here does not fail, it reports the wrong depth's numbers.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from degeneration_probe.data.activation_store import cached_slot_for_probe_layer
from degeneration_probe.probes.linear_probe import LiveMultiLayerProbe

SLOTS, TOKENS, HIDDEN = 6, 4, 8


class _ModelWithLabelledLayers(nn.Module):
    """Hidden-state slot s is filled with the constant s, so a slot is readable."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self._marker = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
        batch = input_ids.shape[0]
        states = tuple(
            torch.full((batch, TOKENS, HIDDEN), float(slot)) for slot in range(SLOTS)
        )
        return SimpleNamespace(hidden_states=states)


class _RecordingHeads(nn.Module):
    """Stands in for the real heads, keeping whatever features it was handed."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, 1)
        self.seen = None

    def forward(self, features):
        self.seen = features
        return {"probe_logits": features.mean(dim=-1)}


def test_a_probed_layer_reads_its_own_slot_and_not_the_embedding():
    """Slot 0 is the embedding, so decoder block L lives at slot L + 1."""
    heads = _RecordingHeads()
    probe = LiveMultiLayerProbe(_ModelWithLabelledLayers(), heads, layers=[0, 2, 4])

    probe(input_ids=torch.zeros(1, TOKENS, dtype=torch.long),
          attention_mask=torch.ones(1, TOKENS, dtype=torch.long))

    # The fixture fills each slot with its index, so the value read back names
    # the slot the depth actually landed on.
    read_back = [float(heads.seen[0, 0, index, 0]) for index in range(3)]
    assert read_back == [1.0, 3.0, 5.0]
    assert read_back == [float(cached_slot_for_probe_layer(l)) for l in (0, 2, 4)]


def test_features_are_stacked_as_the_cached_probe_expects():
    """[batch, tokens, layers, hidden] -- the layout the cached heads take."""
    heads = _RecordingHeads()
    probe = LiveMultiLayerProbe(_ModelWithLabelledLayers(), heads, layers=[1, 3])
    probe(input_ids=torch.zeros(2, TOKENS, dtype=torch.long),
          attention_mask=torch.ones(2, TOKENS, dtype=torch.long))
    assert tuple(heads.seen.shape) == (2, TOKENS, 2, HIDDEN)


def test_a_depth_the_model_cannot_reach_is_refused():
    """Better than silently scoring whatever the last slot happens to be."""
    probe = LiveMultiLayerProbe(_ModelWithLabelledLayers(), _RecordingHeads(), layers=[99])
    with pytest.raises(ValueError, match="hidden-state slot"):
        probe(input_ids=torch.zeros(1, TOKENS, dtype=torch.long),
              attention_mask=torch.ones(1, TOKENS, dtype=torch.long))


def test_scores_are_split_to_the_layer_they_came_from():
    from scripts.score_rollouts import score_split_by_layer

    records = [{
        "prompt_id": "p0", "rollout_idx": 0, "domain": "d", "split": "val",
        "stop_reason": "length", "num_tokens": 3, "onset_position": 1.0,
        "is_positive": True,
        # column j holds a distinct probability, so a misfiled column is visible
        "scores": np.tile(np.array([0.1, 0.5, 0.9], dtype=np.float32), (3, 1)),
    }]

    def fake_records(*args, **kwargs):
        return [dict(record) for record in records]

    import scripts.score_rollouts as sr
    original, sr._score_records = sr._score_records, fake_records
    try:
        frames = score_split_by_layer(
            None, None, None, split="val", metadata=None, batch_size=1, layers=[4, 8, 12]
        )
    finally:
        sr._score_records = original

    assert sorted(frames) == [4, 8, 12]
    for expected, layer in zip([0.1, 0.5, 0.9], [4, 8, 12]):
        assert np.allclose(frames[layer].iloc[0]["scores"], expected)


def test_a_depth_count_that_disagrees_with_the_probe_is_refused():
    from scripts.score_rollouts import score_split_by_layer
    import scripts.score_rollouts as sr

    def fake_records(*args, **kwargs):
        return [{
            "prompt_id": "p0", "rollout_idx": 0, "domain": "d", "split": "val",
            "stop_reason": "length", "num_tokens": 2, "onset_position": None,
            "is_positive": False,
            "scores": np.zeros((2, 2), dtype=np.float32),
        }]

    original, sr._score_records = sr._score_records, fake_records
    try:
        with pytest.raises(ValueError, match="wrong layers"):
            score_split_by_layer(
                None, None, None, split="val", metadata=None, batch_size=1, layers=[1, 2, 3]
            )
    finally:
        sr._score_records = original
