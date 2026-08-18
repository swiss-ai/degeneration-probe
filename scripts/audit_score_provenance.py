"""Say which stored scores know what produced them, and which do not.

A directory of scores is read as one scorer. The thresholds frozen on its
validation split are applied unchanged to every other split beside it, so the
question "which checkpoint wrote this" has to have an answer before any test
split is reported against it.

Scoring records that answer now. Directories written before it did not, and the
answer generally cannot be recovered: a depth's neighbouring checkpoints differ
by less than the precision the replayed metrics carry, and the threshold a
directory froze comes from a coarser grid than the replay's own, so neither is
a fingerprint. What this script can do is narrow it. For a directory with no
recorded provenance it recomputes the selection metrics from the stored scores
and reports which replayed checkpoints sit closest, together with how far the
runner-up is, which is what says whether the nearest match means anything.

    python scripts/audit_score_provenance.py
    python scripts/audit_score_provenance.py --run-dir outputs/<run>/latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.head_selection import validation_record
from degeneration_probe.evaluation.protocol import rollout_score
from degeneration_probe.utils.file_utils import load_json

PROVENANCE_FILE = "scoring_provenance.json"
REPLAY_FILE = "checkpoint_replay.parquet"
# The two quantities a checkpoint is selected on. Matching on both is what makes
# a near miss visible: a directory whose nearest replayed step is barely nearer
# than the next one has not been identified, only guessed at.
MATCH_ON = ["in_pattern_recall", "warning_recall_256"]


def scored_directories(root: Path) -> List[Path]:
    """Every directory holding a validation scores file."""
    return sorted({path.parent.parent for path in root.glob("**/scores/val.parquet")})


def measured(scores_path: Path) -> Optional[Dict[str, float]]:
    """The selection metrics the stored scores imply."""
    frame = pd.read_parquet(scores_path)
    positives = frame[frame["is_positive"].astype(bool)]
    negatives = frame[~frame["is_positive"].astype(bool)]
    if positives.empty or negatives.empty:
        return None
    return validation_record(
        negative_peaks=[rollout_score(s) for s in negatives["scores"]],
        positive_scores=[np.asarray(s, dtype=np.float64) for s in positives["scores"]],
        onsets=[int(o) for o in positives["onset_position"]],
    )


def owning_run(directory: Path) -> Optional[Path]:
    """The run attempt a scores directory belongs to, whatever its depth."""
    for parent in [directory, *directory.parents]:
        if (parent / REPLAY_FILE).is_file():
            return parent
    return None


def candidates(directory: Path, record: Dict[str, float], layer: Optional[int]) -> pd.DataFrame:
    """Replayed checkpoints ranked by how closely they match the stored scores.

    Only a directory scored on the run's own model can be narrowed this way. A
    replay measures every checkpoint against one model's validation answers, so
    comparing it with scores taken on a different model's answers ranks
    checkpoints by a difference in population rather than in weights.
    """
    if any(part == "cross_model" for part in directory.parts):
        return pd.DataFrame()
    run_dir = owning_run(directory)
    if run_dir is None:
        return pd.DataFrame()
    replay = pd.read_parquet(run_dir / REPLAY_FILE)
    if layer is not None:
        replay = replay[replay["layer"] == layer]
    if replay.empty:
        return pd.DataFrame()
    distance = sum((replay[column] - record[column]).abs() for column in MATCH_ON)
    return replay.assign(distance=distance).nsmallest(3, "distance")[
        ["layer", "step", *MATCH_ON, "distance"]
    ]


def layer_of(directory: Path) -> Optional[int]:
    """The depth a directory holds, read from the path the scorer writes to."""
    for part in reversed(directory.parts):
        if part.startswith("layer_") and part[6:].isdigit():
            return int(part[6:])
    return None


def audit(root: Path) -> pd.DataFrame:
    rows = []
    for directory in scored_directories(root):
        provenance = directory / PROVENANCE_FILE
        frozen = (directory / "decision_thresholds.json").is_file()
        if provenance.is_file():
            recorded = load_json(provenance)
            rows.append(
                {
                    "directory": str(directory.relative_to(root)),
                    "provenance": "recorded",
                    "checkpoint": recorded.get("checkpoint"),
                    "layer": recorded.get("layer"),
                    "thresholds frozen": frozen,
                    "nearest replayed step": None,
                    "runner-up gap": None,
                }
            )
            continue
        layer = layer_of(directory)
        record = measured(directory / "scores" / "val.parquet")
        near = candidates(directory, record, layer) if record else pd.DataFrame()
        transferred = any(part == "cross_model" for part in directory.parts)
        if near.empty:
            nearest = gap = None
        else:
            nearest = int(near.iloc[0]["step"])
            gap = (
                round(float(near.iloc[1]["distance"] - near.iloc[0]["distance"]), 6)
                if len(near) > 1
                else None
            )
        rows.append(
            {
                "directory": str(directory.relative_to(root)),
                "provenance": "unknown",
                "checkpoint": None,
                "layer": layer,
                "thresholds frozen": frozen,
                "nearest replayed step": "not comparable" if transferred else nearest,
                "runner-up gap": gap,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="One run attempt. Defaults to every run under outputs/.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write the table as CSV.")
    args = parser.parse_args()

    root = args.run_dir.resolve() if args.run_dir else (REPO_ROOT / "outputs")
    table = audit(root)
    if table.empty:
        print(f"No scored directories under {root}.")
        return

    unknown = int((table["provenance"] == "unknown").sum())
    exposed = int(((table["provenance"] == "unknown") & table["thresholds frozen"]).sum())
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 78)
    print(table.to_string(index=False))
    print(
        f"\n{len(table)} scored directories, {unknown} without recorded provenance, "
        f"{exposed} of those with frozen thresholds beside them."
    )
    if exposed:
        print(
            "\nA directory with frozen thresholds and no provenance is the case to fix "
            "first:\nthe thresholds describe a scorer nobody can name. Re-score it at a "
            "named\ncheckpoint into its own directory before reporting any further split "
            "from it."
        )
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
