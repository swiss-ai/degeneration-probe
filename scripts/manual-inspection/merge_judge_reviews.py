"""Merge per-reviewer manual-inspection exports (see ``export_judge_review.py`` /
``judge_review.html``) into one combined table, and report agreement stats
against the original LLM-judge verdicts.

Each teammate's browser export (``judge_review_<name>_<dataset>.json``,
downloaded via the HTML app's "Esporta le mie revisioni" button) already
carries both the judge's original verdict and the reviewer's correction --
this script just concatenates those files and summarizes them; it does not
need to re-read the pipeline's own parquet outputs.

Usage:
    .venv/bin/python scripts/manual-inspection/merge_judge_reviews.py \\
        --inputs handoff/judge_review_alice_*.json handoff/judge_review_bob_*.json \\
        --out handoff/merged_reviews.parquet

    # or, simpler -- a glob pattern (quoted so the shell doesn't expand it):
    .venv/bin/python scripts/manual-inspection/merge_judge_reviews.py \\
        --inputs "handoff/judge_review_*.json" \\
        --out handoff/merged_reviews.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _expand_inputs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for m in matches:
            p = Path(m)
            if not p.exists():
                raise FileNotFoundError(f"No such file: {p}")
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def load_reviews(paths: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        meta = payload["meta"]
        for review in payload["reviews"]:
            rows.append(
                {
                    "reviewer": meta["reviewer"],
                    "dataset_tag": meta["dataset_tag"],
                    "backend": meta["backend"],
                    "source_file": path.name,
                    **review,
                }
            )
    if not rows:
        raise ValueError("No reviews found in any input file.")
    return pd.DataFrame(rows)


def check_duplicates(df: pd.DataFrame) -> None:
    """Warn (not fail) on the same (id) reviewed more than once -- can happen
    if a reviewer's file is accidentally included twice, or if a row was
    manually reassigned and reviewed by two people. Every copy is kept in the
    merged output; this only flags it so it isn't missed silently."""
    dupes = df[df.duplicated("id", keep=False)].sort_values("id")
    if not dupes.empty:
        n_ids = dupes["id"].nunique()
        print(f"WARNING: {n_ids} id(s) reviewed more than once (all copies kept in output):")
        for _id, group in dupes.groupby("id"):
            reviewers = ", ".join(group["reviewer"].tolist())
            print(f"  {_id}: reviewed by {reviewers}")
        print()


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"Merged {n} review(s) from {df['source_file'].nunique()} file(s), "
          f"{df['reviewer'].nunique()} reviewer(s):")
    print(df.groupby("reviewer").size().to_string())
    print()

    n_classification_corrected = int((df["classification_status"] == "corrected").sum())
    print(f"Classification disagreements (reviewer overrode the judge's is_degenerating): "
          f"{n_classification_corrected}/{n} ({n_classification_corrected / n:.1%})")
    if n_classification_corrected:
        flips = df[df["classification_status"] == "corrected"]
        false_to_true = int(((~flips["judge_is_degenerating"]) & flips["corrected_is_degenerating"]).sum())
        true_to_false = int((flips["judge_is_degenerating"] & (~flips["corrected_is_degenerating"])).sum())
        print(f"  false negative fixed (judge said clean, reviewer says degenerating): {false_to_true}")
        print(f"  false positive fixed (judge said degenerating, reviewer says clean): {true_to_false}")
    print()

    degen = df[df["corrected_is_degenerating"]]
    if not degen.empty:
        pos_counts = degen["position_status"].value_counts()
        print(f"Onset-position review, among the {len(degen)} row(s) judged degenerating "
              f"(after any classification correction):")
        print(pos_counts.to_string())
    print()

    commented = df[df["comment"].fillna("").str.strip() != ""]
    print(f"Rows with a reviewer comment: {len(commented)}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inputs", type=str, nargs="+", required=True,
        help="Paths or glob patterns to reviewer export JSON files (e.g. "
        "'handoff/judge_review_*.json').",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output path for the merged table. Extension picks the format: "
        ".parquet, .csv, or .json.",
    )
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    print(f"Reading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")
    print()

    df = load_reviews(paths)
    check_duplicates(df)
    print_summary(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix == ".json":
        df.to_json(out_path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output extension {suffix!r} -- use .parquet, .csv, or .json")
    print(f"\nWrote merged table ({len(df)} rows) to {out_path}")


if __name__ == "__main__":
    main()
