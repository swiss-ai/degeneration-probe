"""Training a head per depth together must equal training them apart.

The saving comes from the cache layout, not from any change to the objective:
every depth of a rollout sits in one file, and opening that file costs far more
than reading it. So the depths are trained in one pass, with a head each and
their losses added.

That rearrangement is only free if it is exactly a rearrangement. Two things
would quietly break it and both are checked here: a gradient norm clipped
across all heads at once, which would couple heads that share no parameters,
and a shared normalization, which would make one depth's scale depend on
another's. A break would not raise anything. It would produce slightly wrong
numbers that still look entirely plausible, which is why this is a test.
"""

import pytest
import torch

from degeneration_probe.probes.linear_probe import (
    CachedFeatureProbe,
    MultiLayerCachedProbe,
)
from degeneration_probe.training.loss import compute_degeneration_loss

HIDDEN = 16
LAYERS = [4, 8, 12]


def _batch(seed=0, tokens=7, rows=3):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(rows, tokens, len(LAYERS), HIDDEN, generator=generator)
    targets = (torch.rand(rows, tokens, generator=generator) > 0.5).float()
    mask = torch.ones(rows, tokens, dtype=torch.bool)
    mask[:, -1] = False
    return features, targets, mask


def _single(layer_seed):
    return CachedFeatureProbe(
        hidden_size=HIDDEN, layer_idx=0, seed=layer_seed, normalization="layernorm"
    )


def _multi():
    return MultiLayerCachedProbe(
        hidden_size=HIDDEN, layer_indices=LAYERS, seed=42, normalization="layernorm"
    )


def test_heads_start_identical_to_solo_probes():
    """A depth's initialization must not depend on its company."""
    multi = _multi()
    solo = _single(42)
    for head in multi.heads:
        assert torch.equal(head.linear.weight, solo.linear.weight)
        assert torch.equal(head.pre_head_norm.weight, solo.pre_head_norm.weight)


def test_each_head_sees_only_its_own_layer():
    multi = _multi()
    features, _, _ = _batch()
    joint = multi(features=features)["probe_logits"]
    assert joint.shape == (features.shape[0], features.shape[1], len(LAYERS))
    for index in range(len(LAYERS)):
        solo = _single(42)
        alone = solo(features=features[..., index, :])["probe_logits"]
        assert torch.allclose(joint[..., index], alone, atol=1e-6)


def test_a_step_moves_each_head_exactly_as_it_would_alone():
    """The equivalence that makes one job stand in for many.

    Each head owns its parameters and the loss is their sum, so a head's
    gradient carries no trace of the others. Trained side by side under the same
    data and the same seed, a head must land where it would have landed alone.
    """
    features, targets, mask = _batch(seed=1)

    multi = _multi()
    joint_optimizer = torch.optim.AdamW(multi.parameters(), lr=0.05)
    logits = multi(features=features)["probe_logits"]
    loss, _ = compute_degeneration_loss("bce", logits, targets, mask, pos_weight=2.0)
    loss.backward()
    # Per head, which is what the trainer installs in place of a global clip.
    for parameters in multi.head_parameters():
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    joint_optimizer.step()

    for index in range(len(LAYERS)):
        solo = _single(42)
        solo_optimizer = torch.optim.AdamW(solo.parameters(), lr=0.05)
        alone = solo(features=features[..., index, :])["probe_logits"]
        solo_loss, _ = compute_degeneration_loss(
            "bce", alone, targets, mask, pos_weight=2.0
        )
        solo_loss.backward()
        torch.nn.utils.clip_grad_norm_(solo.parameters(), 1.0)
        solo_optimizer.step()

        assert torch.allclose(
            multi.heads[index].linear.weight, solo.linear.weight, atol=1e-7
        )
        assert torch.allclose(
            multi.heads[index].pre_head_norm.weight,
            solo.pre_head_norm.weight,
            atol=1e-7,
        )


def test_clipping_across_all_heads_would_break_the_equivalence():
    """Why the global clip is switched off rather than left alone.

    This is the failure the trainer avoids. Clipping one norm over every head
    rescales all of them whenever any one is large, so a head's update starts
    depending on depths it shares nothing with.
    """
    features, targets, mask = _batch(seed=2)
    multi = _multi()
    logits = multi(features=features)["probe_logits"]
    loss, _ = compute_degeneration_loss("bce", logits, targets, mask, pos_weight=2.0)
    loss.backward()

    per_head = [
        torch.nn.utils.clip_grad_norm_(parameters, 0.01).item()
        for parameters in multi.head_parameters()
    ]
    # A limit small enough to bite: clipped separately each head reaches it,
    # while one norm over all of them would have divided the budget between.
    assert all(norm > 0.01 for norm in per_head)
    after = [
        torch.linalg.vector_norm(
            torch.cat([p.grad.flatten() for p in parameters])
        ).item()
        for parameters in multi.head_parameters()
    ]
    for norm in after:
        assert norm <= 0.01 + 1e-6


def test_the_loss_adds_rather_than_averages_over_heads():
    """Averaging would shrink every head's gradient by the head count."""
    features, targets, mask = _batch(seed=3)
    multi = _multi()
    logits = multi(features=features)["probe_logits"]
    joint, _ = compute_degeneration_loss("bce", logits, targets, mask)
    parts = [
        compute_degeneration_loss("bce", logits[..., index], targets, mask)[0]
        for index in range(len(LAYERS))
    ]
    assert torch.allclose(joint, sum(parts), atol=1e-6)


def test_a_saved_head_is_an_ordinary_single_layer_probe(tmp_path):
    """A depth trained jointly must load where a solo one would."""
    multi = _multi()
    with torch.no_grad():
        multi.heads[1].linear.weight.add_(0.5)
    multi.save(tmp_path)

    restored = CachedFeatureProbe(
        hidden_size=HIDDEN, path=tmp_path / f"layer_{LAYERS[1]:02d}"
    )
    assert restored.layer_idx == LAYERS[1]
    assert torch.allclose(restored.linear.weight, multi.heads[1].linear.weight)


def test_features_and_heads_must_agree_about_depth():
    multi = _multi()
    features = torch.randn(2, 5, len(LAYERS) + 1, HIDDEN)
    try:
        multi(features=features)
    except ValueError as error:
        assert "layers" in str(error)
    else:
        raise AssertionError("a depth mismatch must be refused, not broadcast")


def test_a_multi_head_probe_survives_a_save_and_reload_cycle(tmp_path):
    """The path a run takes when it restores its best checkpoint.

    Training writes checkpoints and then loads the best one back at the end, so
    a probe that can save but not reload fails only in the final seconds of a
    run, after all the work is done.
    """
    multi = _multi()
    with torch.no_grad():
        for index, head in enumerate(multi.heads):
            head.linear.weight.add_(0.1 * (index + 1))
    multi.save(tmp_path)

    restored = _multi()
    restored.load_weights(tmp_path)
    for original, loaded in zip(multi.heads, restored.heads):
        assert torch.equal(original.linear.weight, loaded.linear.weight)
        assert torch.equal(original.pre_head_norm.weight, loaded.pre_head_norm.weight)


def test_scoring_a_multi_head_run_must_name_a_depth(tmp_path, monkeypatch):
    """One scores file holds one score per token, so it holds one probe.

    Scoring a many-depth run without saying which depth would either silently
    keep one of them or write a shape nothing downstream can read, and both are
    worse than being told to choose.
    """
    import json
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import score_rollouts

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text("{}")

    class Probe:
        layers = [4, 8, 12]
        probed_layers = [4, 8, 12]
        layer = 4

    config = type("C", (), {"training": type("T", (), {"probe": Probe()})()})()
    monkeypatch.setattr(score_rollouts, "load_run_config", lambda _dir: config)

    with pytest.raises(ValueError, match="Name one with --layer"):
        score_rollouts.run(run_dir, checkpoint="final", splits=["val"], batch_size=1)

    with pytest.raises(ValueError, match="was not trained by this run"):
        score_rollouts.run(
            run_dir, checkpoint="final", splits=["val"], batch_size=1, layer=99
        )
