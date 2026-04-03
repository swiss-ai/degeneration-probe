"""Vanilla PyTorch training loop for the degeneration probe."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from degeneration_probe.probe import SequenceProbe
from degeneration_probe.evaluation import evaluate

log = logging.getLogger(__name__)


def train(
    probe: SequenceProbe,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    *,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    pos_weight: Optional[float] = None,
    use_wandb: bool = False,
) -> SequenceProbe:
    """
    Train the probe's linear head with BCE loss.

    Only ``probe.linear`` parameters are updated; the base LLM is frozen.

    Args:
        probe: SequenceProbe instance (model already loaded).
        train_loader: DataLoader yielding collated batches.
        eval_loader: Optional validation DataLoader.
        num_epochs: Number of full passes over the training set.
        learning_rate: AdamW learning rate.
        pos_weight: Optional weight for positive class in BCE loss.
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        The trained probe (same object, mutated in-place).
    """
    device = probe.model.device
    optimizer = torch.optim.AdamW(probe.linear.parameters(), lr=learning_rate)

    pw_tensor = None
    if pos_weight is not None:
        pw_tensor = torch.tensor([pos_weight], device=device)

    global_step = 0
    for epoch in range(num_epochs):
        probe.linear.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            completion_mask = batch["completion_mask"].to(device)
            labels = batch["labels"].to(device)

            out = probe(input_ids, attention_mask, completion_mask)
            logit = out["logit"].squeeze(-1)  # [B]

            loss = F.binary_cross_entropy_with_logits(
                logit, labels, pos_weight=pw_tensor,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if use_wandb:
                import wandb
                wandb.log({"train/loss": loss.item(), "train/step": global_step})

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info("Epoch %d/%d — avg train loss: %.4f", epoch + 1, num_epochs, avg_loss)

        if use_wandb:
            import wandb
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        # End-of-epoch evaluation
        if eval_loader is not None:
            metrics = evaluate(probe, eval_loader)
            log.info(
                "  eval — acc: %.3f  f1: %.3f  auc: %.3f",
                metrics["accuracy"],
                metrics["f1"],
                metrics["auc"],
            )
            if use_wandb:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    return probe
