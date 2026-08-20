"""Is the quantity this literature reports saturated at the units it reports on?

The argument the paper makes is that answer-level, chunk-level and sentence-level
detection accuracy cannot rank scorers or choose a depth, because the units are
drawn overwhelmingly from inside a loop that has already started. That is worth
measuring rather than asserting, and it has to be measured at each of the three
granularities, since the nearest prior work does not use one.

Reads stored per-token scores, so it asks the question of any scorer that has been
through the protocol, probe or baseline, and retrains nothing. See
`degeneration_probe/analysis/unit_levels.py` for what a unit is and for the
limits of the correspondence with anybody's actual detector.

    python scripts/unit_level_saturation.py --split val \
      --build-root <build> --tokenizer <tokenizer.json> \
      --scores name=path/to/scores.parquet [name=path ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.analysis.unit_levels import UNITS, measure, segment_split
from degeneration_probe.evaluation.scores import unjudged_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="A tokenizer.json the `tokenizers` library can load directly.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="name=path pairs; the name is what appears in the table. Split at the "
        "LAST '=', so a name may contain one (as 'W=128' does) while a path may not.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip a scorer whose file is absent instead of refusing to run. Off by "
        "default: a silently dropped scorer is a table that is quietly incomplete.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the long-form table. Defaults to "
        "outputs/analysis/unit_level_saturation_<split>.csv",
    )
    args = parser.parse_args()

    out = args.out or (
        REPO_ROOT / "outputs" / "analysis" / f"unit_level_saturation_{args.split}.csv"
    )

    unjudged = unjudged_rollouts(
        args.build_root / "onset_labels" / "onset_labels.parquet"
    )
    print(
        f"Excluding {len(unjudged)} rollouts with no judged outcome, so every scorer "
        "is read on one population",
        flush=True,
    )
    print(f"Segmenting {args.split} into sentences, chunks and answers", flush=True)
    segmented = segment_split(args.build_root, args.tokenizer, args.split)
    unreliable = segmented.pop("__unreliable__")["count"]
    print(f"  {len(segmented)} answers", flush=True)
    if unreliable:
        print(
            f"  warning: {unreliable} answers lost a unit delimiter in the piecewise "
            "decode, so their boundaries are unreliable",
            flush=True,
        )

    requested = []
    for entry in args.scores:
        name, separator, path = entry.rpartition("=")
        if not separator:
            raise SystemExit(f"--scores expects name=path, got {entry!r}")
        requested.append((name, Path(path)))

    absent = [f"{name}: {path}" for name, path in requested if not path.is_file()]
    if absent:
        message = "no score file for " + "; ".join(absent)
        if not args.allow_missing:
            raise SystemExit(
                f"{message}\nPass --allow-missing to report the rest without them."
            )
        print(f"  warning: {message}", flush=True)

    records = []
    for name, path in requested:
        if not path.is_file():
            continue
        for row in measure(path, segmented, unjudged=unjudged):
            records.append({"scorer": name, "split": args.split, **row})
        print(f"  measured {name}", flush=True)

    table = pd.DataFrame(records)
    if table.empty:
        raise SystemExit("No scorer produced any rows.")

    pd.set_option("display.width", 200)
    for kind, _, whose in UNITS:
        subset = table[table["unit"] == kind]
        if subset.empty:
            continue
        print(f"\n=== unit: {kind} (granularity of {whose}) ===")
        shown = subset.drop(columns=["unit", "granularity_of", "split"]).copy()
        for column in ("positive_rate", "auc", "balanced_accuracy"):
            shown[column] = shown[column].round(4)
        print(shown.to_string(index=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
