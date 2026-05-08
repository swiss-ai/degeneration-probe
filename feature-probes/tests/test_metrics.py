import numpy as np
import importlib.util
from pathlib import Path


def _load_metrics_module():
    repo_root = Path(__file__).resolve().parents[1]
    metrics_spec = importlib.util.spec_from_file_location(
        "feature_probes.utils.metrics",
        repo_root / "feature_probes" / "utils" / "metrics.py",
    )
    metrics_module = importlib.util.module_from_spec(metrics_spec)
    metrics_spec.loader.exec_module(metrics_module)
    return metrics_module


compute_thresholded_regression_metrics = _load_metrics_module().compute_thresholded_regression_metrics


def test_thresholded_regression_metrics_uses_configured_threshold():
    labels = np.array([0.2, 0.81, 0.95, 0.4])
    predictions = np.array([0.1, 0.7, 0.9, 0.6])

    metrics = compute_thresholded_regression_metrics(
        predictions=predictions,
        labels=labels,
        threshold=0.8,
    )

    assert metrics["degeneration_threshold"] == 0.8
    assert metrics["degeneration_true_positive_count"] == 2
    assert metrics["degeneration_true_negative_count"] == 2
    assert metrics["degeneration_pred_positive_count"] == 1
    assert metrics["degeneration_pred_negative_count"] == 3
    assert metrics["degeneration_accuracy"] == 0.75
    assert metrics["degeneration_precision"] == 1.0
    assert metrics["degeneration_recall"] == 0.5
    assert metrics["degeneration_f1"] == 2 / 3
    assert metrics["degeneration_positive_rate"] == 0.5
    assert metrics["degeneration_pred_positive_rate"] == 0.25
