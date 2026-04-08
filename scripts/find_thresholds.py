"""Find optimal TTR thresholds for degeneration detection from LLM-labeled data.

Supports comparing across multiple datasets and labeling models.

Example (single file, old format with 'llm_label'):
    uv run python scripts/find_thresholds.py --input_paths outputs/threshold_data.jsonl

Example (multiple files, multi-model labels):
    uv run python scripts/find_thresholds.py \
      --input_paths outputs/alpaca_labeled.jsonl outputs/hermes_labeled.jsonl outputs/aime_labeled.jsonl \
      --label_fields llm_label_gpt_5_4 llm_label_gpt_4o_mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_VALUES = [1, 2, 3]


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dataset_name_from_path(path: Path) -> str:
    """Extract a short dataset name from a file path."""
    stem = path.stem
    for suffix in ("_labeled", "_generations", "_threshold_data"):
        stem = stem.replace(suffix, "")
    return stem


def analyze(records, label_field, metric, n_values):
    """Run threshold analysis for one label field. Returns dict of results per n."""
    labels = []
    for r in records:
        val = r.get(label_field)
        if val is None:
            return None  # field not present
        labels.append(int(val))
    labels = np.array(labels)

    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return None

    flip = metric in ("min_ttr", "mean_ttr")
    results = {}

    for n in n_values:
        scores = []
        for r in records:
            val = r["chunk_summary"]["metrics_by_n"][str(n)][metric]
            scores.append(1.0 - val if flip else val)
        scores = np.array(scores)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_t = thresholds[best_idx]
        display_t = 1.0 - best_t if flip else best_t

        y_pred = (scores >= best_t).astype(int)
        tp = int(((y_pred == 1) & (labels == 1)).sum())
        fp = int(((y_pred == 1) & (labels == 0)).sum())
        fn = int(((y_pred == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[n] = {
            "threshold": display_t,
            "auc": roc_auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_pos": n_pos,
            "n_neg": len(labels) - n_pos,
            "fpr": fpr,
            "tpr": tpr,
            "best_idx": best_idx,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find optimal TTR thresholds for degeneration detection."
    )
    parser.add_argument(
        "--input_paths", type=str, nargs="+", required=True,
        help="One or more labeled JSONL files",
    )
    parser.add_argument(
        "--label_fields", type=str, nargs="+", default=None,
        help="Label field names (default: auto-detect 'llm_label' or 'llm_label_*')",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "outputs" / "threshold_analysis"),
    )
    parser.add_argument(
        "--metric", type=str, default="max_repetition",
        choices=["max_repetition", "mean_repetition", "min_ttr", "mean_ttr"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    datasets = {}
    for p in args.input_paths:
        path = Path(p)
        name = dataset_name_from_path(path)
        datasets[name] = load_records(path)

    # Auto-detect label fields if not specified
    if args.label_fields is None:
        sample = next(iter(datasets.values()))[0]
        label_fields = []
        if "llm_label" in sample:
            label_fields.append("llm_label")
        for key in sorted(sample.keys()):
            if key.startswith("llm_label_"):
                label_fields.append(key)
        if not label_fields:
            print("No label fields found. Run label_generations.py first.")
            return
    else:
        label_fields = args.label_fields

    print(f"Datasets: {list(datasets.keys())}")
    print(f"Label fields: {label_fields}")
    print(f"Metric: {args.metric}\n")

    # Analyze each combination
    all_results = {}
    for ds_name, records in datasets.items():
        for lf in label_fields:
            result = analyze(records, lf, args.metric, N_VALUES)
            if result is None:
                print(f"  {ds_name} / {lf}: skipped (missing field or no class variation)")
                continue
            key = f"{ds_name} / {lf}"
            all_results[key] = result
            n_pos = result[N_VALUES[0]]["n_pos"]
            n_neg = result[N_VALUES[0]]["n_neg"]
            print(f"  {key}: {n_pos} degen, {n_neg} normal")

    if not all_results:
        print("\nNo valid results. Exiting.")
        return

    # Print summary table
    print(f"\n{'Dataset / Label':>40}  {'n':>3}  {'Thr':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("-" * 85)
    for key, result in all_results.items():
        for n in N_VALUES:
            r = result[n]
            print(
                f"{key:>40}  n={n}  {r['threshold']:>6.3f}  {r['auc']:>6.3f}  "
                f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}"
            )

    # Plot: one row per label field, one column per n
    n_rows = len(label_fields)
    n_cols = len(N_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    fig.suptitle(f"ROC Curves — {args.metric}", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    for row, lf in enumerate(label_fields):
        for col, n in enumerate(N_VALUES):
            ax = axes[row][col]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

            for ci, (ds_name, records) in enumerate(datasets.items()):
                key = f"{ds_name} / {lf}"
                if key not in all_results:
                    continue
                r = all_results[key][n]
                ax.plot(
                    r["fpr"], r["tpr"], linewidth=2, color=colors[ci % len(colors)],
                    label=f"{ds_name} (AUC={r['auc']:.3f})",
                )
                ax.scatter(
                    [r["fpr"][r["best_idx"]]], [r["tpr"][r["best_idx"]]],
                    color=colors[ci % len(colors)], zorder=5, s=50,
                )

            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"n={n}  [{lf}]")
            ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plot_path = output_dir / f"roc_comparison_{args.metric}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nROC comparison saved to {plot_path}")

    # Save results as JSON
    json_results = {}
    for key, result in all_results.items():
        json_results[key] = {
            str(n): {k: v for k, v in r.items() if k not in ("fpr", "tpr", "best_idx")}
            for n, r in result.items()
        }
    results_path = output_dir / "threshold_results.json"
    results_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    print(f"Results JSON saved to {results_path}")


if __name__ == "__main__":
    main()
