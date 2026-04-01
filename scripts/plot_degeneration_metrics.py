"""Plot n-gram TTR and repetition distributions from batch generation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "labeled_generations.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots"


def load_records(path: Path) -> list[dict]:
    """Read JSONL generation records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_metric(records: list[dict], n: int, metric_name: str) -> list[float]:
    """Collect aggregated metric values for one n-gram size."""
    values = []
    for record in records:
        metrics_by_n = record.get("chunk_summary", {}).get("metrics_by_n", {})
        metric_block = metrics_by_n.get(str(n))
        if metric_block and metric_name in metric_block:
            values.append(metric_block[metric_name])
    return values


def save_histogram(values: list[float], title: str, xlabel: str, output_path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot chunk-based degeneration metric distributions.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH), help="JSONL file from run_prompt_batch.py")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for saved plots")
    args = parser.parse_args()

    records = load_records(Path(args.input_path))
    output_dir = Path(args.output_dir)

    for n in (1, 3):
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


if __name__ == "__main__":
    main()
