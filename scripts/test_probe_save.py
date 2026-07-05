"""Round-trip tests for ValueHeadProbe.save / load.

Verifies that the fix correctly persists `pre_head_norm` weights alongside
`value_head`, and that backward-compatibility with pre-fix checkpoints is
preserved.

The tests use a mocked tiny model (no real LLM weights) so the script runs
fast and on CPU. We monkey-patch `get_model_hidden_size` / `get_model_layers`
in the `value_head_probe` module namespace before instantiating the probe.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Monkey-patch helpers before ValueHeadProbe uses them
import degeneration_probe.probes.value_head_probe as vhp_module

vhp_module.get_model_hidden_size = lambda m: m._hidden_size  # type: ignore
vhp_module.get_model_layers = lambda m: m._layers  # type: ignore

from degeneration_probe.probes.value_head_probe import ValueHeadProbe  # noqa: E402


class FakeModel(nn.Module):
    """Minimal stub satisfying ValueHeadProbe's expectations on the LLM."""

    def __init__(self, hidden_size: int = 64, n_layers: int = 4, dtype=torch.float32):
        super().__init__()
        self._hidden_size = hidden_size
        self._layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, dtype=dtype) for _ in range(n_layers)]
        )
        self._dtype = dtype

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return self._dtype


def _build_probe_with_known_weights(
    norm_kind: str = "layernorm",
    hidden_size: int = 64,
) -> ValueHeadProbe:
    fake = FakeModel(hidden_size=hidden_size, n_layers=4)
    probe = ValueHeadProbe(
        fake,
        layer_idx=2,
        normalize_before_head=norm_kind,
        probe_dtype="float32",
    )
    with torch.no_grad():
        if norm_kind in ("layernorm", "rmsnorm"):
            probe.pre_head_norm.weight.fill_(2.71828)
            if hasattr(probe.pre_head_norm, "bias") and probe.pre_head_norm.bias is not None:
                probe.pre_head_norm.bias.fill_(3.14159)
        probe.value_head.weight.fill_(1.61803)
        probe.value_head.bias.fill_(-0.5)
    return probe


def test_layernorm_roundtrip():
    """Save → reload preserves trained LayerNorm AND value_head weights."""
    probe = _build_probe_with_known_weights(norm_kind="layernorm")

    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "probe_ln"
        probe.save(save_path)

        # Both files must exist
        assert (save_path / "probe_head.bin").exists(), "probe_head.bin missing"
        assert (save_path / "pre_head_norm.bin").exists(), (
            "pre_head_norm.bin missing — the fix did not write it"
        )
        assert (save_path / "probe_config.json").exists(), "probe_config.json missing"

        # Reload with a fresh fake model
        fake2 = FakeModel(hidden_size=64, n_layers=4)
        probe2 = ValueHeadProbe(
            fake2,
            path=save_path,
            normalize_before_head="layernorm",
            probe_dtype="float32",
        )

        assert torch.allclose(
            probe.pre_head_norm.weight, probe2.pre_head_norm.weight
        ), "LayerNorm.weight not restored"
        assert torch.allclose(
            probe.pre_head_norm.bias, probe2.pre_head_norm.bias
        ), "LayerNorm.bias not restored"
        assert torch.allclose(
            probe.value_head.weight, probe2.value_head.weight
        ), "value_head.weight not restored"
        assert torch.allclose(
            probe.value_head.bias, probe2.value_head.bias
        ), "value_head.bias not restored"
    print("[PASS] LayerNorm round-trip preserves trained weights")


def test_backward_compat_no_norm_file():
    """Old-style checkpoint (no pre_head_norm.bin) still loads with defaults."""
    probe = _build_probe_with_known_weights(norm_kind="layernorm")

    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "probe_old"
        probe.save(save_path)
        # Simulate pre-fix checkpoint by removing the norm file
        (save_path / "pre_head_norm.bin").unlink()

        fake2 = FakeModel(hidden_size=64, n_layers=4)
        probe2 = ValueHeadProbe(
            fake2,
            path=save_path,
            normalize_before_head="layernorm",
            probe_dtype="float32",
        )

        # value_head must still load correctly
        assert torch.allclose(probe.value_head.weight, probe2.value_head.weight)
        assert torch.allclose(probe.value_head.bias, probe2.value_head.bias)
        # pre_head_norm falls back to defaults (weight=1, bias=0)
        assert torch.allclose(
            probe2.pre_head_norm.weight, torch.ones_like(probe2.pre_head_norm.weight)
        ), "LayerNorm.weight should be the default 1.0 when no norm file"
        assert torch.allclose(
            probe2.pre_head_norm.bias, torch.zeros_like(probe2.pre_head_norm.bias)
        ), "LayerNorm.bias should be the default 0.0 when no norm file"
    print("[PASS] Old checkpoint without pre_head_norm.bin loads with defaults")


def test_identity_norm_no_file_written():
    """nn.Identity has empty state_dict → no pre_head_norm.bin should be created."""
    fake = FakeModel(hidden_size=64, n_layers=4)
    probe = ValueHeadProbe(
        fake,
        layer_idx=2,
        normalize_before_head="none",
        probe_dtype="float32",
    )
    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "probe_id"
        probe.save(save_path)
        assert not (save_path / "pre_head_norm.bin").exists(), (
            "Identity must NOT produce pre_head_norm.bin (empty state_dict)"
        )
    print("[PASS] Identity normalization skipped (no pre_head_norm.bin written)")


def test_l2_norm_no_file_written():
    """L2Norm has no params/buffers → no pre_head_norm.bin should be created."""
    fake = FakeModel(hidden_size=64, n_layers=4)
    probe = ValueHeadProbe(
        fake,
        layer_idx=2,
        normalize_before_head="l2",
        probe_dtype="float32",
    )
    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "probe_l2"
        probe.save(save_path)
        assert not (save_path / "pre_head_norm.bin").exists(), (
            "L2Norm must NOT produce pre_head_norm.bin"
        )
    print("[PASS] L2 normalization skipped (no pre_head_norm.bin written)")


def test_rmsnorm_roundtrip():
    """RMSNorm has only weight → round-trip should preserve it."""
    probe = _build_probe_with_known_weights(norm_kind="rmsnorm")

    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "probe_rms"
        probe.save(save_path)
        assert (save_path / "pre_head_norm.bin").exists()

        fake2 = FakeModel(hidden_size=64, n_layers=4)
        probe2 = ValueHeadProbe(
            fake2,
            path=save_path,
            normalize_before_head="rmsnorm",
            probe_dtype="float32",
        )
        assert torch.allclose(probe.pre_head_norm.weight, probe2.pre_head_norm.weight)
    print("[PASS] RMSNorm round-trip preserves weights")


def test_existing_epoch1_inspection():
    """Inspect the actual running-job checkpoint to confirm the bug pre-fix."""
    epoch1 = Path(
        "/capstor/scratch/cscs/lbaggi/feature-probes/probes/"
        "apertus_apertus_stability_repetition_balanced_layer30/epoch_1"
    )
    if not (epoch1 / "probe_head.bin").exists():
        print("[SKIP] epoch_1 checkpoint not available")
        return
    has_norm = (epoch1 / "pre_head_norm.bin").exists()
    print(
        f"[INFO] real epoch_1: pre_head_norm.bin = {has_norm} "
        f"(expected False — saved before the fix)"
    )


if __name__ == "__main__":
    test_layernorm_roundtrip()
    test_rmsnorm_roundtrip()
    test_backward_compat_no_norm_file()
    test_identity_norm_no_file_written()
    test_l2_norm_no_file_written()
    test_existing_epoch1_inspection()
    print("\nAll tests passed.")
