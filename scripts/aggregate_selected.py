"""Gather the per-depth evaluations of every scored run into one table.

Scoring a run at its selected checkpoints leaves one evaluation directory per
depth and split, each holding the four views as separate files. This reads all
of them and writes a flat table with one row per (run, depth, split, budget),
carrying the recipe axes, the selected step and whether the depth was
selectable, the detection figures (precision, recall, accuracy, AUC), the
coverage figures (post-onset and the pre-onset bands at 32, 64, 128 and 256
tokens) and the timing figures (lead, early share, never-fired count).

    python scripts/aggregate_selected.py
    python scripts/aggregate_selected.py --summary     # also print recipe x layer means at 1%

Writes <outputs>/analysis/selected_metrics.csv. A second file,
selected_metrics_by_recipe.csv, averages the seeds of each recipe at each depth
and split, over selectable depths only.

With --human the test splits are read from each depth's evaluation_human
folder (scripts/evaluate_human_labels.py), the human-verified labels, and
validation from evaluation as before, since selection uses validation only.
The two files are then selected_metrics_human.csv and
selected_metrics_human_by_recipe.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs"
)
SPLITS = ["val", "test_indomain", "test_heldout_domains"]
AXES = ["model", "dataset", "features", "selection", "window", "anchor", "label", "horizon", "decay", "lora"]

DETECTION = ["precision", "recall", "f1", "accuracy", "negative_fpr", "positives", "negatives"]
COVERAGE = [
    "in_pattern_recall",
    "warning_recall_32",
    "warning_recall_64",
    "warning_recall_128",
    "warning_recall_256",
    "preonset_coverage_32",
    "preonset_coverage_64",
    "preonset_coverage_128",
    "preonset_coverage_256",
    "token_false_positive_rate",
]
TIMING = ["median_offset", "fired_before_frontier", "never_fired_positives", "detected_positives"]


def recipe_name(axes: Dict) -> str:
    selection = axes.get("selection")
    name = "all_tokens" if selection == "all_tokens" else f"{selection} W={axes.get('window')}"
    if axes.get("anchor"):
        name += f" {axes['anchor']}"
    if axes.get("label") == "frontier_hard":
        name += f" K={axes.get('horizon')}"
    elif axes.get("label") == "frontier_soft":
        name += f" soft {axes.get('decay')}"
    elif axes.get("label") == "token_signal":
        name += f" regression"
    if axes.get("lora") not in (None, "none"):
        name += " +LoRA"
    return name


def read_depth(evaluation: Path, split: str) -> Optional[pd.DataFrame]:
    folder = evaluation / split
    if not (folder / "view_a_detection.csv").is_file():
        return None
    a = pd.read_csv(folder / "view_a_detection.csv")
    b = pd.read_csv(folder / "view_b_coverage.csv")
    c = pd.read_csv(folder / "view_c_lead_time.csv")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rank = summary.get("rank_metrics", {})
    frame = a[["target_negative_fpr", "tau"] + [col for col in DETECTION if col in a.columns]]
    frame = frame.merge(b[["target_negative_fpr"] + [col for col in COVERAGE if col in b.columns]], on="target_negative_fpr", how="left")
    frame = frame.merge(c[["target_negative_fpr"] + [col for col in TIMING if col in c.columns]], on="target_negative_fpr", how="left")
    frame["rollout_auc"] = rank.get("rollout_auc")
    frame["rollout_ap"] = rank.get("rollout_ap")
    frame["persistence"] = summary.get("persistence")
    frame.insert(0, "split", split)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subdir", default="selected")
    parser.add_argument("--summary", action="store_true", help="print recipe x layer means at the 1%% budget")
    parser.add_argument("--human", action="store_true", help="read the test splits on the human-verified labels")
    args = parser.parse_args()
    stem = "selected_metrics_human" if args.human else "selected_metrics"

    rows: List[pd.DataFrame] = []
    incomplete: List[str] = []
    for selection_file in sorted(args.outputs_root.glob(f"*/2*/checkpoint_selection.json")):
        attempt = selection_file.parent
        info = json.loads((attempt / "run_info.json").read_text(encoding="utf-8"))
        axes = info.get("axes", {})
        seed = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("seed:")), None)
        meta = {
            "run_name": info.get("run_name", attempt.parent.name),
            "attempt": attempt.name,
            "recipe": recipe_name(axes),
            "seed": seed,
            **{axis: axes.get(axis) for axis in AXES},
        }
        depths = json.loads(selection_file.read_text(encoding="utf-8"))["depths"]
        for depth in depths:
            layer = int(depth["layer"])
            depth_dir = attempt / args.subdir / f"layer_{layer:02d}"
            for split in SPLITS:
                folder = "evaluation_human" if args.human and split != "val" else "evaluation"
                frame = read_depth(depth_dir / folder, split)
                if frame is None:
                    incomplete.append(f"{meta['run_name']} layer {layer} {split}")
                    continue
                for key, value in meta.items():
                    frame[key] = value
                frame["layer"] = layer
                frame["selected_step"] = int(depth["selected_step"])
                frame["selectable"] = bool(depth["selectable"])
                rows.append(frame)

    if not rows:
        raise SystemExit("no evaluations found yet")
    table = pd.concat(rows, ignore_index=True)
    lead = ["run_name", "attempt", "recipe", "seed"] + AXES + ["layer", "selected_step", "selectable", "split", "target_negative_fpr", "tau"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    out_dir = args.outputs_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{stem}.csv", index=False)

    metrics = [c for c in table.columns if c not in lead and c != "persistence"]
    by_recipe = (
        table[table["selectable"]]
        .groupby(["recipe", "model", "layer", "split", "target_negative_fpr"])[metrics]
        .agg(["mean", "std", "count"])
    )
    by_recipe.columns = [f"{m}_{s}" for m, s in by_recipe.columns]
    by_recipe = by_recipe.reset_index()
    by_recipe.to_csv(out_dir / f"{stem}_by_recipe.csv", index=False)

    runs = table[["run_name", "attempt"]].drop_duplicates()
    print(f"{len(runs)} runs, {len(table)} rows -> {out_dir / f'{stem}.csv'}")
    print(f"recipe x layer means -> {out_dir / f'{stem}_by_recipe.csv'}")
    if incomplete:
        print(f"{len(incomplete)} (run, depth, split) evaluations missing, e.g. {incomplete[0]}")

    if args.summary:
        # The depth is chosen on validation and then read, unchanged, on the test
        # splits: choosing it per split would turn a test split into a second
        # validation set.
        pd.set_option("display.width", 220)
        at_one = by_recipe[by_recipe["target_negative_fpr"] == 0.01]
        chosen = (
            at_one[at_one["split"] == "val"]
            .sort_values("warning_recall_256_mean", ascending=False)
            .groupby(["recipe", "model"])
            .head(1)[["recipe", "model", "layer"]]
        )
        report = at_one.merge(chosen, on=["recipe", "model", "layer"])
        cols = ["warning_recall_256_mean", "warning_recall_256_std", "warning_recall_128_mean", "in_pattern_recall_mean", "fired_before_frontier_mean", "median_offset_mean", "never_fired_positives_mean", "rollout_auc_mean", "warning_recall_256_count"]
        for split in SPLITS:
            part = report[report["split"] == split]
            if part.empty:
                continue
            print(f"\n== {split}, budget 1%, depth chosen on validation per recipe (means over seeds) ==")
            print(part[["recipe", "model", "layer"] + cols].sort_values("warning_recall_256_mean", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
