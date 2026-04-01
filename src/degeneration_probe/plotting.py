"""Histogram plotting of degeneration metric distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt


def collect_metric(records: List[dict[str, Any]], *, n: int, metric_name: str) -> List[float]:
    """Collect one aggregated metric across all generation records."""
    values = []
    for record in records:
        metrics_by_n = record.get("chunk_summary", {}).get("metrics_by_n", {})
        metric_block = metrics_by_n.get(str(n))
        if metric_block and metric_name in metric_block:
            values.append(metric_block[metric_name])
    return values


def save_histogram(values: List[float], *, title: str, xlabel: str, output_path: Path) -> None:
    """Save a histogram to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=20, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metric_distributions(
    records: List[dict[str, Any]],
    *,
    n_values: List[int],
    output_dir: Path,
) -> Path:
    """Plot histogram distributions for chunk TTR and repetition metrics."""
    for n in n_values:
        min_ttr_values = collect_metric(records, n=n, metric_name="min_ttr")
        max_rep_values = collect_metric(records, n=n, metric_name="max_repetition")

        if min_ttr_values:
            save_histogram(
                min_ttr_values,
                title=f"Distribution of min chunk TTR for {n}-grams",
                xlabel="min chunk TTR",
                output_path=output_dir / f"ttr_n{n}.png",
            )
        if max_rep_values:
            save_histogram(
                max_rep_values,
                title=f"Distribution of max chunk repetition for {n}-grams",
                xlabel="max chunk repetition",
                output_path=output_dir / f"repetition_n{n}.png",
            )

    print(f"Saved plots to {output_dir}")
    return output_dir
