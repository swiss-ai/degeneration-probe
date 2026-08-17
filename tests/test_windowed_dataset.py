"""Windows as training items, and the properties the ladder depends on."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from degeneration_probe.config import SelectionConfig
from degeneration_probe.data.activation_store import LAYER_ORDER_LABEL, activation_path
from degeneration_probe.data.dataset import DegenerationRecord
from degeneration_probe.data.windowed_dataset import WindowedActivationDataset

HIDDEN, SLOTS, LAYER = 4, 5, 2
LENGTH, ONSET, W = 80, 50, 8


@pytest.fixture
def cache(tmp_path):
    """Slot s filled with s, and each token's first channel set to its index."""
    for prompt in [f"p{i}" for i in range(6)] + [f"n{i}" for i in range(18)]:
        hidden = torch.stack(
            [torch.full((LENGTH, HIDDEN), float(slot)) for slot in range(SLOTS)]
        )
        hidden[LAYER + 1, :, 0] = torch.arange(LENGTH, dtype=torch.float32)
        path = activation_path(tmp_path, "d", prompt, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"hidden_states": hidden.to(torch.float16)},
            str(path),
            metadata={"layer_order": LAYER_ORDER_LABEL},
        )
    return tmp_path


def _records():
    records = []
    for index in range(6):
        targets = np.zeros(LENGTH, dtype=np.float32)
        targets[ONSET:] = 1.0
        records.append(
            DegenerationRecord(
                prompt_id=f"p{index}", rollout_idx=0, domain="d", split="train",
                prompt_text="", generated_token_ids=np.arange(LENGTH), targets=targets,
                is_positive=True, onset_position=ONSET,
            )
        )
    for index in range(18):
        records.append(
            DegenerationRecord(
                prompt_id=f"n{index}", rollout_idx=0, domain="d", split="train",
                prompt_text="", generated_token_ids=np.arange(LENGTH),
                targets=np.zeros(LENGTH, dtype=np.float32),
            )
        )
    return records


def _dataset(cache, strategy="frontier_window", anchor="centered", **kwargs):
    selection = SelectionConfig(strategy=strategy, window_size=W, anchor=anchor, **kwargs)
    return WindowedActivationDataset(
        _records(), activations_root=str(cache), probe_layer=LAYER,
        selection=selection, batch_size=4, seed=7,
    )


def test_an_item_is_a_window_of_the_probed_layer_at_the_right_positions(cache):
    dataset = _dataset(cache)
    item = dataset[0]
    assert item["features"].shape == (W, HIDDEN)
    # Channel 0 carries the token index, so the window's positions are readable.
    positions = item["features"][:, 0].tolist()
    assert positions == sorted(positions)
    assert len(set(positions)) == W
    # And the states come from the probed layer, not a neighbour.
    assert torch.all(item["features"][:, 1] == LAYER + 1)


def test_targets_follow_the_same_positions_as_the_features(cache):
    dataset = _dataset(cache)
    for index in range(len(dataset)):
        item = dataset[index]
        positions = item["features"][:, 0].long()
        expected = (positions >= ONSET).float()
        if item["prompt_id"].startswith("p"):
            assert torch.equal(item["targets"], expected)
        else:
            assert torch.all(item["targets"] == 0)


def test_reading_the_dataset_in_order_reproduces_the_composed_batches(cache):
    dataset = _dataset(cache)
    batch_size = 4
    for start in range(0, len(dataset) - batch_size + 1, batch_size):
        batch = [dataset[start + offset] for offset in range(batch_size)]
        positives = sum(item["prompt_id"].startswith("p") for item in batch)
        assert positives >= 1


def test_resampling_moves_the_windows_and_keeps_the_budget(cache):
    dataset = _dataset(cache, strategy="random_window")
    before = [dataset[i]["features"][:, 0].tolist() for i in range(len(dataset))]
    dataset.resample(1)
    after = [dataset[i]["features"][:, 0].tolist() for i in range(len(dataset))]
    assert before != after
    assert all(len(window) == W for window in after)


def test_the_summary_describes_what_training_sees_not_the_pool(cache):
    exhaustive = _dataset(cache, strategy="all_tokens").summary()
    anchored = _dataset(cache, anchor="centered").summary()
    assert exhaustive["valid_tokens"] == 24 * LENGTH
    # One window per rollout, so the budget is fixed rather than length-driven.
    assert anchored["valid_tokens"] <= 24 * W
    # Anchoring concentrates the budget on the boundary, so the positive share
    # rises even though far fewer tokens are seen.
    assert anchored["positive_token_rate"] > exhaustive["positive_token_rate"]


def test_a_rule_that_excludes_the_positive_class_is_refused(cache):
    # A trailing window is entirely run-up, and a horizon-0 label marks none of
    # it, so the pairing would train on no positive token at all.
    with pytest.raises(ValueError, match="selected no positive tokens"):
        _dataset(cache, anchor="trailing")
    # The same window with a label that reaches back into the run-up is fine.
    records = _records()
    for index, record in enumerate(records[:6]):
        targets = np.asarray(record.targets).copy()
        targets[ONSET - 4 :] = 1.0  # a horizon of four tokens
        records[index] = record.__class__(**{**record.__dict__, "targets": targets})
    dataset = WindowedActivationDataset(
        records, activations_root=str(cache), probe_layer=LAYER,
        selection=SelectionConfig(strategy="frontier_window", window_size=W, anchor="trailing"),
        batch_size=4, seed=7,
    )
    assert dataset.summary()["positive_tokens"] > 0


def test_every_rung_is_reachable_through_the_config(cache):
    for strategy in ("all_tokens", "rollout_balanced", "random_window", "frontier_window"):
        dataset = _dataset(cache, strategy=strategy)
        assert len(dataset) > 0
        assert dataset[0]["features"].shape[1] == HIDDEN
