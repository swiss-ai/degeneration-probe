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
