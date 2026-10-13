"""Where a scorer's flags fall relative to the onset, at a fixed false-alarm budget.

Warning coverage over a window and pre-onset coverage before it summarise the
same quantity at two distances. This reports it at every distance: for each bin
of distance before the onset (and a few bins after it), the share of tokens of
degenerate responses that the scorer flags at its validation threshold, pooled
over responses and averaged over seeds. A scorer that reads the approaching
loop has a profile that rises toward the onset; one that flags degenerate
responses from their start has a flat one.

    python scripts/onset_profile.py --scorer "apertus frozen L15" <depth dir> [...] \\
        --scorer ... --split test_indomain --budget 0.05 --out <csv>

Each depth directory holds scores/<split>.parquet and decision_thresholds.json;
several directories under one scorer are its seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import read_scores  # noqa: E402

BEFORE = [(lo, lo + 32) for lo in range(0, 1024, 32)]
AFTER = [(0, 32), (32, 128), (128, 512), (512, 4096)]


def profile(depth: Path, split: str, budget: float) -> pd.DataFrame:
    thresholds = json.loads((depth / "decision_thresholds.json").read_text())["thresholds"]
    tau = next(t["tau"] for t in thresholds if abs(t["target_negative_fpr"] - budget) < 1e-9)
    frame = read_scores(depth / "scores" / f"{split}.parquet", validate=False)
    frame = frame[frame["is_positive"].astype(bool)]
    edges = BEFORE + [(1024, None)]
    flagged = {("before", e): [0, 0] for e in edges}
    flagged.update({("after", e): [0, 0] for e in AFTER})
    for scores, onset in zip(frame["scores"], frame["onset_position"].astype(int)):
        above = np.asarray(scores, dtype=np.float64) >= tau
        for lo, hi in edges:
            start = 0 if hi is None else max(0, onset - hi)
            stop = max(0, onset - lo) if hi is not None else max(0, onset - 1024)
            span = above[start:stop]
            flagged[("before", (lo, hi))][0] += int(span.sum())
            flagged[("before", (lo, hi))][1] += span.size
        for lo, hi in AFTER:
            span = above[onset + lo : onset + hi]
            flagged[("after", (lo, hi))][0] += int(span.sum())
            flagged[("after", (lo, hi))][1] += span.size
    rows = []
    for (side, (lo, hi)), (hits, total) in flagged.items():
        label = f"{lo}-{hi}" if hi is not None else f">{lo}"
        rows.append({"side": side, "bin": label, "lo": lo, "flag_rate": hits / total if total else np.nan, "tokens": total})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer", nargs="+", action="append", required=True, metavar=("NAME", "DIR"))
    parser.add_argument("--split", default="test_indomain")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for name, *dirs in args.scorer:
        for seed, depth in enumerate(dirs):
            part = profile(Path(depth), args.split, args.budget)
            part["scorer"], part["seed"] = name, seed
            frames.append(part)
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = table.groupby(["scorer", "side", "lo", "bin"], sort=False)["flag_rate"].mean().reset_index()
    print(summary.pivot_table(index=["side", "lo", "bin"], columns="scorer", values="flag_rate", sort=False).round(3).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
