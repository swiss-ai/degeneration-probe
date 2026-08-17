"""The cached regime must agree with the live one, and refuse to lie.

Reading stored activations instead of running the model is only sound while
nothing adapts the model, and only useful if the head behaves identically
either way. Both properties fail silently if broken: a probe trained on stale
features still converges, and a diverged head still produces plausible scores.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from types import SimpleNamespace

import degeneration_probe.probes.linear_probe as probe_module
from degeneration_probe.config import (
    FeaturesConfig,
    LabelConfig,
    LoraConfig,
    ProbeConfig,
    TrainingConfig,
)
from degeneration_probe.data.activation_store import LAYER_ORDER_LABEL, activation_path
from degeneration_probe.data.cached_dataset import CachedActivationDataset, cached_collate_fn
from degeneration_probe.data.dataset import DegenerationRecord
from degeneration_probe.probes.linear_probe import (
    CachedFeatureProbe,
    DegenerationProbe,
    setup_cached_probe,
)

HIDDEN, SLOTS, TOKENS = 4, 5, 6
LAYER = 2


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, HIDDEN)
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(SLOTS)])

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(probe_module, "get_model_layers", lambda model: list(model.layers))
    monkeypatch.setattr(probe_module, "get_model_hidden_size", lambda model: HIDDEN)


@pytest.mark.parametrize("context_window", [1, 3])
def test_a_checkpoint_behaves_the_same_in_either_regime(patched, tmp_path, context_window):
    live = DegenerationProbe(
        TinyModel(), layer_idx=LAYER, context_window_size=context_window, normalization="layernorm"
    )
    with torch.no_grad():
        live.linear.weight.normal_(std=0.5)
        live.linear.bias.fill_(0.25)
    live.save(tmp_path)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    live_logits = live(input_ids=input_ids)["probe_logits"]
    states = live._captured

    # The same weights, applied to the states the live pass just produced.
    cached = CachedFeatureProbe(hidden_size=HIDDEN, path=tmp_path)
    cached_logits = cached(features=states.detach())["probe_logits"]

    assert cached.context_window_size == context_window
    assert torch.allclose(live_logits, cached_logits, atol=1e-5)


def test_the_cached_probe_reads_its_shape_from_the_checkpoint(patched, tmp_path):
    live = DegenerationProbe(TinyModel(), layer_idx=LAYER, context_window_size=2)
    live.save(tmp_path)
    cached = setup_cached_probe(
        ProbeConfig(id="p", layer=LAYER, context_window_size=2), hidden_size=HIDDEN,
        checkpoint_path=tmp_path,
    )
    assert cached.layer_idx == LAYER
    assert torch.allclose(cached.linear.weight, live.linear.weight)


def test_training_on_stale_features_is_refused_rather_than_silently_wrong():
    with pytest.raises(ValueError, match="cannot train adapters"):
        TrainingConfig(
            task="degeneration",
            short_name="d",
            probe=ProbeConfig(id="p"),
            features=FeaturesConfig(regime="cached"),
            lora=LoraConfig(enabled=True, layers="all"),
        )
    # Without adapters the combination is sound.
    accepted = TrainingConfig(
        task="degeneration",
        short_name="d",
        probe=ProbeConfig(id="p"),
        features=FeaturesConfig(regime="cached"),
        lora=LoraConfig(enabled=False),
    )
    assert accepted.features.regime == "cached"


def _write_cache(root, domain, prompt_id, rollout_idx, tokens):
    hidden = torch.stack(
        [torch.full((tokens, HIDDEN), float(slot)) for slot in range(SLOTS)]
    ).to(torch.float16)
    path = activation_path(root, domain, prompt_id, rollout_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": hidden}, str(path), metadata={"layer_order": LAYER_ORDER_LABEL})


def test_the_dataset_pairs_the_probed_layer_with_its_targets(tmp_path):
    _write_cache(tmp_path, "d", "p0", 0, TOKENS)
    record = DegenerationRecord(
        prompt_id="p0",
        rollout_idx=0,
        domain="d",
        split="train",
        prompt_text="",
        generated_token_ids=np.arange(TOKENS),
        targets=np.array([0, 0, 0, 1, 1, 1], dtype=np.float32),
    )
    tokenization = SimpleNamespace(max_completion_length=4096)
    dataset = CachedActivationDataset(
        [record], activations_root=str(tmp_path), probe_layer=LAYER, tokenization=tokenization
    )
    item = dataset[0]
    assert item["features"].shape == (TOKENS, HIDDEN)
    # Probed layer 2 is cached slot 3, and the fixture fills each slot with its index.
    assert torch.all(item["features"] == 3.0)
    assert item["targets"].tolist() == [0, 0, 0, 1, 1, 1]
    assert dataset.summary()["positive_tokens"] == 3


def test_batches_pad_to_the_longest_rollout_and_mask_what_was_added(tmp_path):
    for index, tokens in enumerate((3, 6)):
        _write_cache(tmp_path, "d", f"p{index}", 0, tokens)
    records = [
        DegenerationRecord(
            prompt_id=f"p{index}",
            rollout_idx=0,
            domain="d",
            split="train",
            prompt_text="",
            generated_token_ids=np.arange(tokens),
            targets=np.zeros(tokens, dtype=np.float32),
        )
        for index, tokens in enumerate((3, 6))
    ]
    tokenization = SimpleNamespace(max_completion_length=4096)
    dataset = CachedActivationDataset(
        records, activations_root=str(tmp_path), probe_layer=LAYER, tokenization=tokenization
    )
    batch = cached_collate_fn([dataset[0], dataset[1]])
    assert batch["features"].shape == (2, 6, HIDDEN)
    assert batch["target_mask"][0].tolist() == [True] * 3 + [False] * 3
    assert batch["target_mask"][1].all()
