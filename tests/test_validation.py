import numpy as np
import pytest
import torch

from degeneration_probe.evaluation.evaluate import evaluate_probe


class FixedProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, attention_mask=None):
        return {"probe_logits": self.anchor + torch.zeros_like(input_ids, dtype=torch.float32)}


def _negative_batch():
    """A healthy answer, which is what a false-alarm budget is measured on."""
    batch = _batch("val")
    batch["prompt_id"] = ["n"]
    batch["targets"] = torch.tensor([[0.0, 0.0, float("nan")]])
    batch["is_positive"] = [False]
    batch["onset_position"] = [None]
    return batch


def _batch(split):
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "targets": torch.tensor([[0.0, 1.0, float("nan")]]),
        "target_mask": torch.tensor([[True, True, False]]),
        "prompt_id": ["p"],
        "rollout_idx": [0],
        "domain": ["d"],
        "split": [split],
    }


@pytest.mark.parametrize("split", ["val", "test_indomain", "test_heldout_domains"])
def test_evaluation_keys_for_each_final_split(split):
    metrics = evaluate_probe(
        FixedProbe(),
        [_batch(split)],
        loss_name="bce",
        prefix=split,
        pos_weight=2.0,
        metric_names=[],
    )
    assert set(
        [
            f"{split}/loss",
            f"{split}/valid_tokens",
            f"{split}/target_mean",
            f"{split}/prediction_mean",
            f"{split}/prediction_std",
            f"{split}/pos_weight",
        ]
    ).issubset(metrics)
    assert metrics[f"{split}/valid_tokens"] == 2
    # The loss is reported once, under a name that does not change with the loss.
    assert f"{split}/bce_loss" not in metrics


def test_an_unweighted_loss_is_reported_beside_the_weighted_one():
    """The class weight belongs to one recipe, so it cannot carry a comparison.

    Every recipe weights its own training stream differently, which rescales the
    monitoring loss without the model changing. The unweighted loss is measured
    identically for all of them.
    """
    weighted = evaluate_probe(
        FixedProbe(), [_batch("val")], loss_name="bce", prefix="val", pos_weight=8.0
    )
    lightly = evaluate_probe(
        FixedProbe(), [_batch("val")], loss_name="bce", prefix="val", pos_weight=2.0
    )
    # Two weights, one model: the weighted losses disagree, the plain ones do not.
    assert weighted["val/loss"] != pytest.approx(lightly["val/loss"])
    assert weighted["val/loss_unweighted"] == pytest.approx(lightly["val/loss_unweighted"])
    # Weighting a half-positive split upward can only raise the loss.
    assert weighted["val/loss"] > weighted["val/loss_unweighted"]


def test_the_unweighted_loss_matches_the_weighted_one_when_unweighted():
    for weight in (None, 1.0):
        metrics = evaluate_probe(
            FixedProbe(), [_batch("val")], loss_name="bce", prefix="val", pos_weight=weight
        )
        assert metrics["val/loss"] == pytest.approx(metrics["val/loss_unweighted"])


def test_the_monitor_reports_the_rule_record_rather_than_a_ranking():
    """What validation measures is what the checkpoint rule reads.

    A ranking saturates: separating a mostly-degenerate answer from a healthy
    one is easy, so it reaches its ceiling long before the probe is useful. So
    does rollout-level recall on a split with a hundred degenerate answers.
    Coverage of the tokens before the loop keeps moving after both have stopped,
    and it is what the project is about, so it is what a run is steered by.
    """
    from degeneration_probe.evaluation.evaluate import monitor_record

    # Ninety healthy answers spread low, and one the probe is confidently wrong
    # about, which is what sets the bar when only a one percent false-alarm rate
    # is affordable.
    negatives = [0.1 + 0.004 * index for index in range(90)] + [0.99]
    # Ten degenerate answers, each flagged from ten tokens before its loop.
    positives = [np.concatenate([np.full(40, 0.2), np.full(60, 0.995)]) for _ in range(10)]
    onsets = [50] * 10

    record = monitor_record(negatives, positives, onsets)
    assert record["budget_realized_fpr"] <= 0.01 + 1e-9
    assert record["budget_tau"] > 0.9
    # Every degenerate answer is caught, and caught before its loop.
    assert record["recall_at_budget"] == pytest.approx(1.0)
    assert record["never_fired_positives"] == 0
    assert record["median_offset"] == pytest.approx(-10.0)
    # Coverage inside the loop is complete; coverage of the run-up is the ten
    # flagged tokens out of the band, which is the number that keeps moving.
    assert record["in_pattern_recall"] == pytest.approx(1.0)
    assert record["warning_recall_256"] == pytest.approx(10 / 50)


def test_the_record_is_empty_without_both_populations():
    """A threshold needs healthy answers and coverage needs degenerate ones."""
    from degeneration_probe.evaluation.evaluate import monitor_record

    assert monitor_record([], [np.full(4, 0.9)], [2]) == {}
    assert monitor_record([0.5], [], []) == {}


def test_a_depth_below_the_floor_ranks_under_every_depth_above_it():
    """The gate is what stops an accidental early alarm from being selected.

    A depth that has not learned to flag the loop can still post run-up coverage,
    by firing more or less at random. Ranking it on that would select a
    checkpoint that saw nothing, so a depth only enters the ranking once its
    coverage inside the loop clears the floor.
    """
    from degeneration_probe.evaluation.evaluate import selection_score
    from degeneration_probe.evaluation.head_selection import StoppingRule

    rule = StoppingRule(floor=0.3, band=256)
    blind = {"in_pattern_recall": 0.05, "warning_recall_256": 0.9}
    seeing = {"in_pattern_recall": 0.5, "warning_recall_256": 0.02}
    assert selection_score(blind, rule) < selection_score(seeing, rule)
    assert selection_score(seeing, rule) == pytest.approx(0.02)
    # No rule and no record are both "nothing to say", not a score of zero.
    assert selection_score(seeing, None) is None
    assert selection_score({}, rule) is None


def test_evaluation_reports_the_record_when_the_batch_carries_a_frontier():
    """The frontier has to survive the collate, or none of this is measurable."""
    from degeneration_probe.evaluation.head_selection import StoppingRule

    batch = _batch("val")
    batch["is_positive"] = [True]
    batch["onset_position"] = [1]
    metrics = evaluate_probe(
        FixedProbe(),
        [batch, _negative_batch()],
        loss_name="bce",
        prefix="val",
        rule=StoppingRule(),
    )
    assert "val/in_pattern_recall" in metrics
    assert "val/warning_recall_256" in metrics
    assert "val/selection_score" in metrics
    assert metrics["val/measured_positive_rollouts"] == 1


def test_a_frontier_past_the_scored_completion_is_left_out():
    """A truncated answer whose loop starts past the cap has nothing to measure.

    Counting it as covered or as missed would both be wrong: the tokens the
    claim would be about were never scored.
    """
    from degeneration_probe.evaluation.head_selection import StoppingRule

    batch = _batch("val")
    batch["is_positive"] = [True]
    # Two tokens are scored; the loop is recorded as starting well past them.
    batch["onset_position"] = [900]
    metrics = evaluate_probe(
        FixedProbe(),
        [batch, _negative_batch()],
        loss_name="bce",
        prefix="val",
        rule=StoppingRule(),
    )
    assert metrics["val/positive_rollouts"] == 1
    assert metrics["val/measured_positive_rollouts"] == 0
    assert "val/warning_recall_256" not in metrics
