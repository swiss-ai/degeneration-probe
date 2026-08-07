import torch
import torch.nn.functional as F

from degeneration_probe.training.loss import compute_bce_loss, compute_mse_loss


def test_bce_uses_pos_weight_and_masks_inactive_tokens():
    logits = torch.tensor([[0.0, 1.0, -3.0]])
    targets = torch.tensor([[0.0, 1.0, float("nan")]])
    mask = torch.tensor([[True, True, True]])
    loss, active = compute_bce_loss(logits, targets, mask, pos_weight=3.0)
    expected = F.binary_cross_entropy_with_logits(
        logits[0, :2], targets[0, :2], pos_weight=torch.tensor(3.0)
    )
    assert active == 2
    assert torch.allclose(loss, expected)


def test_mse_applies_sigmoid_and_masks_nan_and_padding():
    logits = torch.tensor([[0.0, 2.0, -2.0, 8.0]])
    targets = torch.tensor([[0.25, 0.75, float("nan"), 0.1]])
    mask = torch.tensor([[True, True, True, False]])
    loss, active = compute_mse_loss(logits, targets, mask)
    expected = ((torch.sigmoid(logits[0, :2]) - targets[0, :2]) ** 2).mean()
    assert active == 2
    assert torch.allclose(loss, expected)
