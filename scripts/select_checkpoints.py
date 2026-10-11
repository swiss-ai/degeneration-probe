"""Pick one checkpoint per depth of every replayed run, and say whether it counts.

The replay (``scripts/replay_checkpoints.py``) measures every saved checkpoint of
every depth on the whole validation split. This turns that table into one
decision per (run, depth), under the rule the stopping logic also uses, with one
addition: a depth that never clears the post-onset floor still gets a step.

    selectable depth   post-onset coverage reached the floor at some step:
                       the step with the highest pre-onset coverage among
                       the steps at or above the floor (earliest on a tie).
    unselectable depth the floor was never reached: the step with the highest
                       post-onset coverage, so the depth can still be scored
                       and reported, flagged ``selectable = False``.

Every depth of every run therefore has a step to score, and the flag records
whether that depth may compete for "best layer" afterwards.

    python scripts/select_checkpoints.py
    python scripts/select_checkpoints.py --pairs   # one "layer:step ..." line per run

Reads <outputs>/<run>/<attempt>/checkpoint_replay.parquet, writes a row per depth
to <outputs>/analysis/checkpoint_selection.csv and a per-run
checkpoint_selection.json beside each replay.
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
REPLAY_FILE = "checkpoint_replay.parquet"
SELECTION_FILE = "checkpoint_selection.json"
GATE = "in_pattern_recall"

AXES = ["model", "dataset", "features", "selection", "window", "anchor", "label", "horizon", "decay", "signal", "loss", "pos_weight", "lora", "seed"]


def select_depth(trajectory: pd.DataFrame, *, floor: float, objective: str) -> Dict:
    steps = trajectory.sort_values("step")
    eligible = steps[steps[GATE] >= floor]
    if len(eligible):
        pool, key, selectable = eligible, objective, True
    else:
        pool, key, selectable = steps, GATE, False
    # idxmax returns the first maximum, which after the sort is the earliest step.
    chosen = pool.loc[pool[key].idxmax()]
    record = {
        "selected_step": int(chosen["step"]),
        "selectable": selectable,
        "best_in_pattern": float(steps[GATE].max()),
        "evaluations": int(len(steps)),
    }
    for column in steps.columns:
        if column not in ("step", "layer"):
            record[column] = chosen[column]
    return record


def load_run(attempt_dir: Path) -> Optional[Dict]:
    info_path = attempt_dir / "run_info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    axes = info.get("axes", {})
    seed = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("seed:")), None)
    return {
        "run_name": info.get("run_name", attempt_dir.parent.name),
        "group": info.get("group"),
        "attempt": attempt_dir.name,
        "status": info.get("status"),
        **{axis: axes.get(axis) for axis in AXES if axis != "seed"},
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--floor", type=float, default=0.3, help="post-onset coverage a depth must reach")
    parser.add_argument("--band", type=int, default=256, help="pre-onset band the objective is measured over")
    parser.add_argument("--out", type=Path, default=None, help="default <outputs-root>/analysis/checkpoint_selection.csv")
    parser.add_argument("--pairs", action="store_true", help="also print one 'layer:step ...' line per run")
    parser.add_argument("--only-latest", action="store_true", help="one attempt per run: the one 'latest' points at")
    args = parser.parse_args()

    objective = f"warning_recall_{args.band}"
    rows: List[Dict] = []
    missing: List[str] = []
    for run_root in sorted(p for p in args.outputs_root.iterdir() if p.is_dir() and p.name != "analysis"):
        attempts = sorted(p for p in run_root.glob("2*") if p.is_dir())
        if args.only_latest and (run_root / "latest").exists():
            attempts = [(run_root / "latest").resolve()]
        for attempt in attempts:
            replay = attempt / REPLAY_FILE
            if not replay.is_file():
                missing.append(f"{run_root.name}/{attempt.name}")
                continue
            meta = load_run(attempt)
            if meta is None:
                continue
            history = pd.read_parquet(replay)
            if objective not in history.columns or GATE not in history.columns:
                raise KeyError(f"{replay} lacks {objective!r} or {GATE!r}")
            per_run = []
            for layer, trajectory in history.groupby("layer"):
                decision = select_depth(trajectory, floor=args.floor, objective=objective)
                row = {**meta, "layer": int(layer), **decision}
                rows.append(row)
                per_run.append({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()})
            (attempt / SELECTION_FILE).write_text(
                json.dumps({"floor": args.floor, "objective": objective, "depths": per_run}, indent=2, default=str),
                encoding="utf-8",
            )

    table = pd.DataFrame(rows)
    out = args.out or args.outputs_root / "analysis" / "checkpoint_selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    runs = table[["run_name", "attempt"]].drop_duplicates() if len(table) else []
    print(f"{len(runs)} replayed runs, {len(table)} depths -> {out}")
    if len(table):
        print(f"  selectable depths: {int(table['selectable'].sum())} / {len(table)}")
        print(table.groupby("selectable")["layer"].agg(["count", "min", "max"]).to_string())
    if missing:
        print(f"{len(missing)} attempts without a replay (not selected):")
        for name in missing:
            print(f"  {name}")
    if args.pairs and len(table):
        for (run, attempt), group in table.groupby(["run_name", "attempt"]):
            pairs = " ".join(f"{int(r.layer)}:{int(r.selected_step)}" for r in group.sort_values("layer").itertuples())
            print(f"{run}/{attempt} {pairs}")


if __name__ == "__main__":
    main()
