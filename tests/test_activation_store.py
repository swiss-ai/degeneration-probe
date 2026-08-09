import pytest
import torch
from safetensors.torch import save_file

from degeneration_probe.data.activation_store import (
    LAYER_ORDER_LABEL,
    activation_path,
    agrees_with_live_capture,
    cached_slot_for_probe_layer,
    load_probe_layer,
    probe_layer_for_cached_slot,
)

SLOTS, TOKENS, HIDDEN = 5, 7, 4


@pytest.fixture
def build_root(tmp_path):
    """A cache whose slot s is filled with the constant s, so the slot is readable."""
    hidden = torch.stack(
        [torch.full((TOKENS, HIDDEN), float(slot)) for slot in range(SLOTS)]
    ).to(torch.float16)
    path = activation_path(tmp_path, "d", "p0", 3)
    path.parent.mkdir(parents=True)
    save_file(
        {"hidden_states": hidden},
        str(path),
        metadata={"layer_order": LAYER_ORDER_LABEL, "completion_len": str(TOKENS)},
    )
    return tmp_path


def test_the_embedding_slot_shifts_every_layer_by_one():
    assert cached_slot_for_probe_layer(0) == 1
    assert cached_slot_for_probe_layer(30) == 31
    assert probe_layer_for_cached_slot(31) == 30
    with pytest.raises(ValueError):
        cached_slot_for_probe_layer(-1)
    with pytest.raises(ValueError, match="embedding output"):
        probe_layer_for_cached_slot(0)


def test_reading_a_probe_layer_lands_on_the_block_and_not_the_embedding(build_root):
    layer = load_probe_layer(build_root, "d", "p0", 3, probe_layer=2)
    assert layer.shape == (TOKENS, HIDDEN)
    # Slot 3 holds block 2, which is what probe layer 2 must return.
    assert torch.equal(layer.float(), torch.full((TOKENS, HIDDEN), 3.0))


def test_a_layer_past_the_end_is_refused_rather_than_wrapped(build_root):
    with pytest.raises(ValueError, match="decoder blocks"):
        load_probe_layer(build_root, "d", "p0", 3, probe_layer=SLOTS)


def test_a_token_count_that_disagrees_with_the_rollout_is_refused(build_root):
    with pytest.raises(ValueError, match="tokens"):
        load_probe_layer(build_root, "d", "p0", 3, probe_layer=1, expected_tokens=TOKENS + 1)


def test_a_different_layer_ordering_is_refused_rather_than_assumed(tmp_path):
    path = activation_path(tmp_path, "d", "p0", 0)
    path.parent.mkdir(parents=True)
    save_file(
        {"hidden_states": torch.zeros(SLOTS, TOKENS, HIDDEN, dtype=torch.float16)},
        str(path),
        metadata={"layer_order": "blocks_only"},
    )
    with pytest.raises(ValueError, match="layer order"):
        load_probe_layer(tmp_path, "d", "p0", 0, probe_layer=1)


def test_a_missing_rollout_names_what_was_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="d/absent/1"):
        load_probe_layer(tmp_path, "d", "absent", 1, probe_layer=1)


def test_a_cached_layer_can_be_checked_against_a_live_capture():
    cached = torch.randn(TOKENS, HIDDEN, dtype=torch.float16)
    assert agrees_with_live_capture(cached, cached.to(torch.bfloat16))
    assert not agrees_with_live_capture(cached, cached + 1.0)
    assert not agrees_with_live_capture(cached, cached[:-1])
