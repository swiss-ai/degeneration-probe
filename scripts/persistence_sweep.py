"""Re-run the protocol at several persistence windows, from stored scores.

The alarm rule fires only where ``m`` consecutive tokens clear the threshold.
For each window the threshold is recalibrated on validation so that every row
spends the same rollout-level false-alarm budget, the first alarm is read at
the token that completes the first run of ``m`` flags, post-onset and
pre-onset coverage are read as plain token coverage at that threshold, and the
test splits are read against the thresholds frozen on validation.

Plain coverage counts every token at or above the lowered threshold, including
tokens that never belong to a run of ``m`` and so never move the alarm. The
alarm-rule coverage next to it counts only the tokens on which a run of ``m``
flags closes, the tokens at which the rule itself has fired.

    python scripts/persistence_sweep.py \\
        --probe "apertus frozen" outputs/<run>/<attempt>/selected/layer_15 [more seeds] \\
        --probe "apertus adapted" ... \\
        --out outputs/analysis/persistence_sweep.csv

Several directories under one name are seeds and are averaged in the summary.
Which rollouts the judge could not rule on is read from the run configuration
beside each scores directory, so probes of different builds can share a call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.protocol import (  # noqa: E402
    choose_thresholds,
    coverage_window,
    per_rollout_table,
    persistence_scores,
    rank_metrics,
    view_a_detection,
    view_b_coverage,
    view_c_lead_time,
)
from degeneration_probe.evaluation.scores import read_scores, unjudged_rollouts  # noqa: E402
from scripts.evaluate_scores import _discover_onset_labels  # noqa: E402

TUNING_SPLIT = "val"
SPLITS = ["val", "test_indomain", "test_heldout_domains"]
BANDS = (32, 64, 128, 256)


def alarm_rule_coverage(frame: pd.DataFrame, thresholds, persistence: int) -> Dict[float, Dict[str, float]]:
    """Coverage of the tokens on which a run of m flags closes, pooled over positive rollouts."""
    keys = ["in_pattern", *(f"warning_{w}" for w in BANDS)]
    hits = {t.target_negative_fpr: dict.fromkeys(keys, 0) for t in thresholds}
    total = dict.fromkeys(keys, 0)
    for row in frame.itertuples(index=False):
        if not row.is_positive:
            continue
        scores = np.asarray(row.scores, dtype=np.float64)
        closing = np.r_[np.full(min(persistence - 1, scores.size), -np.inf), persistence_scores(scores, persistence)]
        onset = int(row.onset_position)
        segments = {"in_pattern": closing[onset:],
                    **{f"warning_{w}": closing[coverage_window(scores.size, onset, w)] for w in BANDS}}
        for key, segment in segments.items():
            total[key] += segment.size
            for t in thresholds:
                hits[t.target_negative_fpr][key] += int(np.count_nonzero(segment >= t.tau))
    return {k: {f"alarm_{key}": (h[key] / total[key] if total[key] else np.nan) for key in keys}
            for k, h in hits.items()}


def measure(val: pd.DataFrame, target: pd.DataFrame, budgets, persistence: int) -> List[Dict[str, float]]:
    thresholds = choose_thresholds(val, budgets, persistence=persistence)
    table = per_rollout_table(target, thresholds, persistence=persistence)
    a = view_a_detection(table).set_index("target_negative_fpr")
    b = view_b_coverage(target, thresholds).set_index("target_negative_fpr")
    c = view_c_lead_time(table).set_index("target_negative_fpr")
    auc = rank_metrics(table)["rollout_auc"]
    alarm = alarm_rule_coverage(target, thresholds, persistence)
    rows = []
    for t in thresholds:
        k = t.target_negative_fpr
        rows.append({
            "target_negative_fpr": k,
            "tau": float(t.tau),
            "realized_negative_fpr": float(t.realized_negative_fpr),
            "saturated": bool(t.saturated),
            "rollout_auc": auc,
            "precision": float(a.loc[k, "precision"]),
            "recall": float(a.loc[k, "recall"]),
            "accuracy": float(a.loc[k, "accuracy"]),
            "negative_fpr": float(a.loc[k, "negative_fpr"]),
            "in_pattern_recall": float(b.loc[k, "in_pattern_recall"]),
            **{f"warning_recall_{w}": float(b.loc[k, f"warning_recall_{w}"]) for w in BANDS},
            **{f"preonset_coverage_{w}": float(b.loc[k, f"preonset_coverage_{w}"]) for w in BANDS},
            **alarm[k],
            "fired_before_frontier": float(c.loc[k, "fired_before_frontier"]),
            "median_offset": float(c.loc[k, "median_offset"]),
            "never_fired_positives": int(c.loc[k, "never_fired_positives"]),
            "detected_positives": int(c.loc[k, "detected_positives"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", nargs="+", action="append", metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--persistence", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--budgets", nargs="*", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: List[dict] = []
    for spec in args.probe:
        name, dirs = spec[0], spec[1:]
        for seed_index, scores_dir in enumerate(dirs):
            scores_dir = Path(scores_dir).resolve()
            labels = _discover_onset_labels(scores_dir)
            if labels is None:
                raise SystemExit(f"{scores_dir}: cannot find the onset table of its build")
            unjudged = unjudged_rollouts(labels)
            val = read_scores(scores_dir / "scores" / f"{TUNING_SPLIT}.parquet", split=TUNING_SPLIT, unjudged=unjudged)
            targets = {
                split: val if split == TUNING_SPLIT else read_scores(
                    scores_dir / "scores" / f"{split}.parquet", split=split, unjudged=unjudged
                )
                for split in args.splits
            }
            for m in args.persistence:
                for split, target in targets.items():
                    for row in measure(val, target, args.budgets, m):
                        rows.append({"probe": name, "seed_index": seed_index, "scores_dir": str(scores_dir),
                                     "persistence": m, "split": split, **row})
                latest = [r for r in rows if r["probe"] == name and r["seed_index"] == seed_index
                          and r["persistence"] == m and r["target_negative_fpr"] == args.budgets[0]]
                print(f"{name:<24} seed {seed_index} m={m:<2}: "
                      + ", ".join(f"{r['split'][:4]} Warn256 {r['warning_recall_256']:.3f}" for r in latest), flush=True)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    metrics = [c for c in table.columns if c not in ("probe", "seed_index", "scores_dir", "persistence", "split", "target_negative_fpr", "saturated")]
    summary = table.groupby(["probe", "split", "target_negative_fpr", "persistence"])[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"))
    print(f"\n{len(table)} rows -> {args.out}")
    pd.set_option("display.width", 220)
    show = summary.reset_index()
    show = show[show["split"] == "test_indomain"]
    print(show[["probe", "target_negative_fpr", "persistence", "in_pattern_recall_mean", "warning_recall_256_mean",
                "preonset_coverage_256_mean", "median_offset_mean", "fired_before_frontier_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
