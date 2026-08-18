"""The identity the replay of saved checkpoints rests on.

Replaying is only cheap because a trained layer normalization followed by a
linear map is itself a linear map, so a whole run's checkpoints collapse to one
matrix and are applied to states that were read once. If that identity ever
stopped holding, the replay would still produce a full table of plausible
numbers describing a probe that was never trained, so it is worth pinning.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from degeneration_probe.config import ProbeConfig
from degeneration_probe.probes.linear_probe import setup_cached_probe
from replay_checkpoints import check_collapsible, collapse_heads, score_rollout

HIDDEN = 12
LAYERS = [4, 12]


def _probe(**overrides):
    config = ProbeConfig(id="p", layers=LAYERS, normalization="layernorm", **overrides)
    return setup_cached_probe(config, hidden_size=HIDDEN, seed=0, device=torch.device("cpu"))


def _trained(probe):
    """Move the head off its initialization, so a fold is worth testing."""
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.copy_(torch.rand(parameter.shape, generator=generator) - 0.5)
    return probe


def test_a_collapsed_checkpoint_scores_exactly_as_the_probe_does(tmp_path):
    probe = _trained(_probe())
    checkpoint = tmp_path / "checkpoint-50"
    probe.save(checkpoint)

    tokens = 7
    features = torch.randn(len(LAYERS), tokens, HIDDEN, generator=torch.Generator().manual_seed(2))

    weights, biases = collapse_heads(tmp_path, [50], LAYERS)
    assert weights.shape == (len(LAYERS), 1, HIDDEN)
    replayed = score_rollout(features, weights, biases)[0]

    # The cache stores [depths, tokens, hidden]; the probe reads a batch of
    # [tokens, depths, hidden].
    live = torch.sigmoid(
        probe(features=features.permute(1, 0, 2).unsqueeze(0))["probe_logits"]
    )
    # [batch, tokens, depths] to [depths, tokens].
    assert torch.allclose(replayed, live[0].t(), atol=1e-5)


def test_the_width_comes_from_the_checkpoint(tmp_path):
    """A model of another size replays as it trained, not as a constant says."""
    _trained(_probe()).save(tmp_path / "checkpoint-50")
    weights, _ = collapse_heads(tmp_path, [50], LAYERS)
    assert weights.shape[-1] == HIDDEN


@pytest.mark.parametrize(
    "overrides, expected",
    [({"normalization": "rmsnorm"}, "layernorm"), ({"context_window_size": 3}, "one state")],
)
def test_a_run_the_identity_does_not_describe_is_refused(overrides, expected):
    """Better a refusal than a table of numbers belonging to no probe."""
    from types import SimpleNamespace

    fields = {"normalization": "layernorm", "context_window_size": 1, **overrides}
    config = SimpleNamespace(training=SimpleNamespace(probe=SimpleNamespace(**fields)))
    with pytest.raises(SystemExit, match=expected):
        check_collapsible(config)
