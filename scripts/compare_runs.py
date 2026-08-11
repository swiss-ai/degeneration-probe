"""Pool seed repeats and read one recipe against another.

    python scripts/compare_runs.py --root outputs --split test_indomain
    python scripts/compare_runs.py --root outputs --ladder <group> <group> ...

Without a ladder it reports every recipe pooled over its seeds. With one it
also reports the adjacent differences, each beside the spread it came from,
since a difference smaller than that spread is not a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from degeneration_probe.analysis.run_comparison import (
    collect_results,
    collect_runs,
    ladder_deltas,
    pool_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--split", default="test_indomain")
    parser.add_argument("--ladder", nargs="*", default=None, help="Group names, in rung order.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = collect_runs(args.root)
    if runs.empty:
        raise SystemExit(f"No runs under {args.root}")
    finished = runs[runs["status"] == "finished"]
    results = collect_results(finished, args.split)
    if results.empty:
        raise SystemExit(
            f"No run under {args.root} has protocol output for split {args.split!r}. "
            "Run score_rollouts.py then evaluate_scores.py first."
        )

    pooled = pool_seeds(results)
    pd.set_option("display.width", 200)
    print(f"=== {args.split}: recipes pooled over their seeds ===")
    columns = ["group", "target_negative_fpr", "seeds"] + [
        c for c in pooled.columns if c.endswith(("_mean", "_std")) and "precision" in c or
        c in ("recall_mean", "recall_std", "median_offset_mean", "median_offset_std")
    ]
    print(pooled[[c for c in columns if c in pooled.columns]].round(4).to_string(index=False))

    if args.ladder:
        deltas = ladder_deltas(pooled, args.ladder)
        if len(deltas):
            print("\n=== adjacent rungs, each against the spread it came from ===")
            show = ["from", "to", "target_negative_fpr", "seeds"] + [
                c for c in deltas.columns
                if c.startswith(("precision_", "recall_", "median_offset_"))
            ]
            print(deltas[[c for c in show if c in deltas.columns]].round(4).to_string(index=False))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(args.out / f"pooled_{args.split}.csv", index=False)
        if args.ladder:
            ladder_deltas(pooled, args.ladder).to_csv(
                args.out / f"ladder_{args.split}.csv", index=False
            )
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
