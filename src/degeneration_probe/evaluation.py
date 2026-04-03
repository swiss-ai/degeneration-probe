"""Evaluation pipeline for the degeneration probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from degeneration_probe.probe import SequenceProbe
from degeneration_probe.utils.clf_metrics import (
    compute_clf_metrics,
    plot_roc_curve,
    plot_threshold_analysis,
)


@torch.no_grad()
def evaluate(
    probe: SequenceProbe,
    loader: DataLoader,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run the probe on *loader* and return classification metrics.

    Args:
        probe: Trained SequenceProbe.
        loader: DataLoader yielding collated batches.
        threshold: Decision threshold for binary predictions.

    Returns:
        Dictionary of metric name → value.
    """
    device = probe.model.device
    probe.linear.eval()

    all_probs: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        completion_mask = batch["completion_mask"].to(device)
        labels = batch["labels"]

        out = probe(input_ids, attention_mask, completion_mask)
        probs = torch.sigmoid(out["logit"].squeeze(-1)).cpu()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    preds_np = (probs_np >= threshold).astype(float)

    return compute_clf_metrics(preds_np, labels_np, probs_np)


def evaluate_and_save(
    probe: SequenceProbe,
    loader: DataLoader,
    output_dir: str | Path,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the probe, save metrics JSON and plots to *output_dir*.

    Returns:
        The metrics dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect predictions
    device = probe.model.device
    probe.linear.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            completion_mask = batch["completion_mask"].to(device)

            out = probe(input_ids, attention_mask, completion_mask)
            probs = torch.sigmoid(out["logit"].squeeze(-1)).cpu()

            all_probs.extend(probs.tolist())
            all_labels.extend(batch["labels"].tolist())

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    preds_np = (probs_np >= threshold).astype(float)

    metrics = compute_clf_metrics(preds_np, labels_np, probs_np)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    plot_roc_curve(labels_np, probs_np, str(output_dir / "roc_curve.png"))
    plot_threshold_analysis(labels_np, probs_np, str(output_dir / "threshold_analysis.png"))

    return metrics
