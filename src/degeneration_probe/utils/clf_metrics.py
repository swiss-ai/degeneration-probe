"""Classification metrics and ROC curve utilities for probe evaluation."""

import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_clf_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        preds: Binary predictions (0 or 1).
        labels: Ground truth labels (0 or 1).
        probs: Probability scores (continuous).

    Returns:
        Dictionary of metric name → value.
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    has_both_classes = len(np.unique(labels)) == 2
    auc_score = roc_auc_score(labels, probs) if has_both_classes else float("nan")

    optimal_threshold = 0.5
    threshold_optimized_accuracy = float("nan")
    recall_at_01_fpr = float("nan")

    if has_both_classes:
        fpr, tpr, _ = roc_curve(labels, probs)

        # Find optimal threshold for accuracy via grid search
        unique_probs = np.unique(probs)
        candidates = (
            np.percentile(unique_probs, np.linspace(0, 100, 100))
            if len(unique_probs) > 100
            else unique_probs
        )
        best_acc = 0.0
        for t in candidates:
            acc = accuracy_score(labels, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                optimal_threshold = float(t)
        threshold_optimized_accuracy = best_acc

        # Recall at 0.1 FPR
        idx = np.where(fpr <= 0.1)[0]
        recall_at_01_fpr = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_score),
        "optimal_threshold": float(optimal_threshold),
        "threshold_optimized_accuracy": float(threshold_optimized_accuracy),
        "recall_at_0.1_fpr": float(recall_at_01_fpr),
        "total_samples": len(labels),
        "positive_count": int(np.sum(labels == 1.0)),
        "negative_count": int(np.sum(labels == 0.0)),
    }


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str,
) -> None:
    """Plot and save a single ROC curve."""
    if len(np.unique(labels)) < 2:
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.fill_between(fpr, tpr, color="#f9c97d", alpha=0.5)
    plt.plot(fpr, tpr, lw=2, color="black", label=f"ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    for fpr_target in (0.05, 0.1, 0.2):
        idx = np.argmin(np.abs(fpr - fpr_target))
        plt.scatter(fpr[idx], tpr[idx], s=40, color="black", zorder=5)
        plt.text(fpr[idx] + 0.02, tpr[idx], f"{tpr[idx]:.3f}", fontsize=9)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str,
) -> None:
    """Plot accuracy, precision, and recall as a function of classification threshold."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    thresholds = np.linspace(0, 1, 100)
    accuracies, precisions, recalls = [], [], []
    for t in thresholds:
        p = (probs >= t).astype(int)
        accuracies.append(accuracy_score(labels, p))
        precisions.append(precision_score(labels, p, zero_division=0))
        recalls.append(recall_score(labels, p, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs Classification Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
