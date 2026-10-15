"""Response-level AUC and best accuracy of every evaluated scorer, in one table.

The aggregated metric table carries the AUC of each scored depth but not its
best accuracy, the highest accuracy any threshold on the response score reaches.
Both are read here from the per-rollout table each evaluation leaves behind,
which holds the response score of every rollout after the rollouts the judge
could not rule on have been removed. The population is therefore the one every
other number of that evaluation was computed on.

Scorers covered: every depth under <outputs>/*/<attempt>/selected/layer_NN and
every baseline under <outputs>/baselines/<dataset>/<name>.

    python scripts/response_metrics_pool.py              # val and test_indomain
    python scripts/response_metrics_pool.py --workers 32 --splits val test_indomain test_heldout_domains

Writes <outputs>/analysis/response_metrics_pool.csv. A rerun reads only the
tables that changed since the last one. With --human the test splits are read
from each scorer's evaluation_human folder, the human-verified labels, and the
table is response_metrics_pool_human.csv.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.analysis.unit_levels import best_accuracy  # noqa: E402

DEFAULT_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs"
)
COLUMNS = ["prompt_id", "rollout_idx", "is_positive", "rollout_score"]


def measure(path: str) -> Dict:
    from sklearn.metrics import roc_auc_score

    table = pd.read_csv(path, usecols=COLUMNS).drop_duplicates(["prompt_id", "rollout_idx"])
    labels = table["is_positive"].astype(bool).to_numpy()
    scores = table["rollout_score"].astype(float).fillna(0.0).to_numpy()
    both = labels.any() and not labels.all()
    return {
        "rollout_table": path,
        "mtime_ns": Path(path).stat().st_mtime_ns,
        "rollouts": int(labels.size),
        "positives": int(labels.sum()),
        "auc": float(roc_auc_score(labels, scores)) if both else np.nan,
        "best_accuracy": best_accuracy(labels.astype(int), scores) if both else np.nan,
    }


def discover(root: Path, splits: List[str], human: bool = False) -> List[Path]:
    found = []
    for split in splits:
        folder = "evaluation_human" if human and split != "val" else "evaluation"
        found += root.glob(f"*/2*/selected/layer_*/{folder}/{split}/rollout_table.csv")
        found += root.glob(f"baselines/*/*/{folder}/{split}/rollout_table.csv")
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="*", default=["val", "test_indomain"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--human", action="store_true", help="read the test splits on the human-verified labels")
    args = parser.parse_args()

    out = args.outputs_root / "analysis" / ("response_metrics_pool_human.csv" if args.human else "response_metrics_pool.csv")
    previous = pd.read_csv(out) if out.is_file() else pd.DataFrame(columns=["rollout_table", "mtime_ns"])
    known = dict(zip(previous["rollout_table"], previous["mtime_ns"]))

    paths = discover(args.outputs_root, args.splits, args.human)
    stale = [str(p) for p in paths if known.get(str(p)) != p.stat().st_mtime_ns]
    print(f"{len(paths)} rollout tables, {len(stale)} new or changed")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fresh = pd.DataFrame(list(pool.map(measure, stale, chunksize=8)))

    kept = previous[previous["rollout_table"].isin([str(p) for p in paths])]
    if not fresh.empty:
        kept = kept[~kept["rollout_table"].isin(fresh["rollout_table"])]
    table = pd.concat([kept, fresh], ignore_index=True)

    # Identify each row by the depth directory and split it describes, which is
    # what readers join on.
    tables = table["rollout_table"].map(Path)
    table["split"] = [p.parent.name for p in tables]
    table["scorer_dir"] = [str(p.parents[2]) for p in tables]
    table = table.sort_values(["scorer_dir", "split"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"{len(table)} rows -> {out}")


if __name__ == "__main__":
    main()
