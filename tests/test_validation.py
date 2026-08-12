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


def test_the_monitor_reports_recall_at_a_fixed_false_alarm_budget():
    """A rank metric saturates; the operating point is what a run is steered by.

    Separating a mostly-degenerate rollout from a healthy one is easy, so ranking
    reaches its ceiling long before a probe is useful. Recall under a cap on
    false alarms keeps moving after that, which is what makes it worth selecting
    on.
    """
    from degeneration_probe.evaluation.evaluate import operating_point

    # Ninety negatives spread low, ten positives above all but one of them.
    peaks = {("n", i): 0.1 + 0.004 * i for i in range(90)}
    labels = {("n", i): False for i in range(90)}
    # One negative the probe is confidently wrong about, which is what sets the
    # bar when only a one percent false-alarm rate is affordable.
    peaks[("n", 99)] = 0.99
    labels[("n", 99)] = False
    for i in range(10):
        peaks[("p", i)] = 0.7 + 0.01 * i
        labels[("p", i)] = True

    point = operating_point(peaks, labels)
    assert point["budget_realized_fpr"] <= 0.01 + 1e-9
    assert 0.0 <= point["recall_at_budget"] <= 1.0
    # The one confidently wrong negative sets the bar, so it costs recall.
    assert point["budget_tau"] > 0.9


def test_the_operating_point_is_undefined_without_both_classes():
    from degeneration_probe.evaluation.evaluate import operating_point

    assert operating_point({("a", 0): 0.5}, {("a", 0): True}) == {}
    assert operating_point({("a", 0): 0.5}, {("a", 0): False}) == {}
