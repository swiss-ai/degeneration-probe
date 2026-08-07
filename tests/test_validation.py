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
            f"{split}/bce_loss",
            f"{split}/valid_tokens",
            f"{split}/target_mean",
            f"{split}/prediction_mean",
            f"{split}/pos_weight",
        ]
    ).issubset(metrics)
    assert metrics[f"{split}/valid_tokens"] == 2
