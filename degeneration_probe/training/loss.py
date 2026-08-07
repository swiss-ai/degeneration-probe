"""Masked token-level losses for the degeneration task."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _valid_mask(targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    return target_mask.bool() & torch.isfinite(targets)


def compute_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    pos_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """Compute masked ``BCEWithLogitsLoss`` on binary per-token targets."""
    valid = _valid_mask(targets, target_mask)
    active_tokens = int(valid.sum().item())
    if not active_tokens:
        return logits.sum() * 0.0, 0
    selected_targets = targets[valid].float()
    if not torch.all((selected_targets == 0) | (selected_targets == 1)):
        raise ValueError("BCE targets must be exactly 0 or 1")
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(float(pos_weight), device=logits.device, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(
        logits[valid].float(), selected_targets, pos_weight=weight, reduction="mean"
    )
    return loss, active_tokens


def compute_mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Compute masked MSE between ``sigmoid(logit)`` and repetition score."""
    valid = _valid_mask(targets, target_mask)
    active_tokens = int(valid.sum().item())
    if not active_tokens:
        return logits.sum() * 0.0, 0
    predictions = torch.sigmoid(logits[valid].float())
    loss = F.mse_loss(predictions, targets[valid].float(), reduction="mean")
    return loss, active_tokens


def compute_degeneration_loss(
    loss_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    pos_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    if loss_name == "bce":
        return compute_bce_loss(
            logits, targets, target_mask, pos_weight=pos_weight
        )
    if loss_name == "mse":
        return compute_mse_loss(logits, targets, target_mask)
    raise ValueError(f"Unknown degeneration loss {loss_name!r}")
