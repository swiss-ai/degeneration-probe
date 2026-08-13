"""Pooling seed repeats, and reading one recipe against another.

A single run is a point estimate. Two recipes differ by some amount, and
without knowing how much a recipe varies from seed to seed there is no way to
say whether that amount means anything. So every comparison here is made over a
group: the runs that share a recipe and differ only in their seed, which is
exactly what the group label a run records identifies.

A ladder is then read as adjacent deltas. Each rung changes one decision
relative to the one before it, so the difference between adjacent rungs is
attributable to that decision, and is reported against the spread of the two
groups it came from rather than on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Metrics worth pooling: the detection quality at an operating point, what it
# cost, and when the alarm arrived.
POOLED_METRICS = [
    "precision",
    "recall",
    "f1",
    "negative_fpr",
    "median_offset",
    "never_fired_positives",
    "false_early_stop_rate",
    "in_pattern_recall",
    "token_false_positive_rate",
]


def collect_runs(root: Path) -> pd.DataFrame:
    """Every attempt under a root, with its identity and its axes."""
    rows = []
    for info_path in sorted(Path(root).glob("*/*/run_info.json")):
        if info_path.parent.is_symlink():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        rows.append(
            {
                "run_dir": str(info_path.parent),
                "run_name": info.get("run_name"),
                "group": info.get("group"),
                "status": info.get("status"),
                "seed": (info.get("axes") or {}).get("seed"),
                **{f"axis.{k}": v for k, v in (info.get("axes") or {}).items()},
            }
        )
    return pd.DataFrame(rows)


def evaluation_dir(run_dir: Path, split: str, layer: Optional[int], own_layer=None) -> Optional[Path]:
    """Where one depth of one run keeps its protocol output, if it has any.

    A run carrying a probe at every depth stores each one separately, since a
    score file holds one score per token and cannot represent several probes at
    once. A run that trained a single depth keeps its output at the root, and
    can only answer for the depth it trained.
    """
    root = Path(run_dir)
    if layer is not None:
        scoped = root / "layers" / f"layer_{int(layer):02d}" / "evaluation" / split
        if scoped.is_dir():
            return scoped
        if own_layer is not None and int(own_layer) != int(layer):
            return None
    plain = root / "evaluation" / split
    return plain if plain.is_dir() else None


def collect_results(runs: pd.DataFrame, split: str, layer: Optional[int] = None) -> pd.DataFrame:
    """The protocol's views for each run, joined into one tidy frame.

    Naming a layer reads that depth of every run. Leaving it out reads whatever
    each run stored at its root, which is what a single-depth run writes.
    """
    frames = []
    # Records rather than tuples: the axis columns carry dots in their names,
    # which a named tuple cannot hold and would silently rename.
    for row in runs.to_dict("records"):
        base = evaluation_dir(Path(row["run_dir"]), split, layer, row.get("axis.layer"))
        if base is None:
            continue
        detection = base / "view_a_detection.csv"
        if not detection.is_file():
            continue
        frame = pd.read_csv(detection)
        for view, columns in (
            ("view_c_lead_time", ["median_offset", "never_fired_positives", "false_early_stop_rate"]),
            ("view_b_coverage", ["in_pattern_recall", "token_false_positive_rate"]),
        ):
            path = base / f"{view}.csv"
            if path.is_file():
                extra = pd.read_csv(path)
                keep = ["target_negative_fpr"] + [c for c in columns if c in extra.columns]
                frame = frame.merge(extra[keep], on="target_negative_fpr", how="left")
        summary = base / "summary.json"
        if summary.is_file():
            rank = json.loads(summary.read_text(encoding="utf-8")).get("rank_metrics", {})
            for key, value in rank.items():
                frame[key] = value
        frame["run_dir"] = row["run_dir"]
        frame["group"] = row["group"]
        frame["seed"] = row["seed"]
        frame["layer"] = layer if layer is not None else row.get("axis.layer")
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pool_seeds(results: pd.DataFrame, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Mean and spread across the seed repeats of each recipe.

    A metric reported without its spread invites reading noise as an effect,
    which with a hundred-odd positives is easy to do.
    """
    if results.empty:
        return results
    metrics = [m for m in (metrics or POOLED_METRICS + ["rollout_auc", "rollout_ap"])
               if m in results.columns]
    grouped = results.groupby(["group", "split", "target_negative_fpr"], sort=True)
    pooled = grouped[metrics].agg(["mean", "std", "count"])
    pooled.columns = [
        f"{metric}_{statistic}" for metric, statistic in pooled.columns
    ]
    pooled = pooled.reset_index()
    pooled["seeds"] = grouped["seed"].nunique().values
    return pooled


def ladder_deltas(
    pooled: pd.DataFrame,
    order: Sequence[str],
    *,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Adjacent-rung differences, each against the spread it came from.

    Only adjacent rungs are compared, because only they differ by one decision.
    The spread reported beside a delta is the two groups' variation added in
    quadrature, which is the scale a difference has to beat to mean anything.
    """
    if pooled.empty:
        return pooled
    metrics = [m for m in (metrics or POOLED_METRICS) if f"{m}_mean" in pooled.columns]
    indexed = pooled.set_index(["group", "split", "target_negative_fpr"])
    rows = []
    for lower, upper in zip(order, order[1:]):
        for split, budget in {
            (split, budget)
            for group, split, budget in indexed.index
            if group in (lower, upper)
        }:
            if (lower, split, budget) not in indexed.index:
                continue
            if (upper, split, budget) not in indexed.index:
                continue
            before = indexed.loc[(lower, split, budget)]
            after = indexed.loc[(upper, split, budget)]
            record = {
                "from": lower,
                "to": upper,
                "split": split,
                "target_negative_fpr": budget,
                "seeds": int(min(before.get("seeds", 1), after.get("seeds", 1))),
            }
            for metric in metrics:
                delta = after[f"{metric}_mean"] - before[f"{metric}_mean"]
                spread = float(
                    np.hypot(
                        np.nan_to_num(before.get(f"{metric}_std", np.nan)),
                        np.nan_to_num(after.get(f"{metric}_std", np.nan)),
                    )
                )
                record[f"{metric}_delta"] = delta
                record[f"{metric}_spread"] = spread
                # A delta smaller than the spread it sits in is not a result.
                record[f"{metric}_beats_noise"] = bool(
                    spread > 0 and abs(delta) > spread
                )
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["split", "target_negative_fpr", "from"])
