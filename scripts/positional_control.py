"""The positional control: does a scorer's number depend on where its flags sit?

A response-level score pooled with the maximum is invariant to the order of
the token scores it pools, so any transform that keeps each rollout's multiset
of scores leaves the response score, its ranking, the calibrated threshold,
and therefore AUC and accuracy untouched. The onset-aligned numbers are not
invariant. Two such transforms are applied to every scorer and the whole
protocol is re-run on each:

    shuffled      the scores of a rollout permuted among its tokens
    argmax only   the highest score kept at its position, every other token
                  set to the floor of the score range

Requiring ``m`` consecutive tokens above the threshold (persistence) makes the
response score order-dependent again, so each transform is also run at several
persistence windows, with the threshold recalibrated at each so that every row
spends the same false-alarm budget. A scorer whose evidence is a single token
cannot produce ``m`` flags in a row and collapses to chance; a scorer that
flags a large share of a rollout still does after shuffling.

The token-cap baseline, which scores the final token of a capped answer and
nothing else, is built from the first scorer's table and reported alongside.

    python scripts/positional_control.py \\
        --scorer "best probe by AUC" outputs/<run>/<attempt>/selected/layer_15 [more dirs] \\
        --scorer "best probe, adapted" ... \\
        --out outputs/analysis/positional_control.csv

Thresholds are always frozen on validation; ``--split`` names the split that
is then read against them. Several directories under one name are seeds and
are averaged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.protocol import (  # noqa: E402
    choose_thresholds,
    per_rollout_table,
    rank_metrics,
    view_a_detection,
    view_b_coverage,
    view_c_lead_time,
)
from degeneration_probe.evaluation.scores import read_scores  # noqa: E402

TUNING_SPLIT = "val"
SEED = 0


def as_scored(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return frame


def shuffled(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    out["scores"] = [rng.permutation(np.asarray(s)) for s in frame["scores"]]
    return out


def argmax_only(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    floor = float(min(np.asarray(s).min() for s in frame["scores"] if len(s)))
    out = frame.copy()
    kept = []
    for s in frame["scores"]:
        s = np.asarray(s, dtype=np.float32)
        t = np.full_like(s, floor)
        if s.size:
            t[int(np.argmax(s))] = s.max()
        kept.append(t)
    out["scores"] = kept
    return out


TRANSFORMS: Dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = {
    "as scored": as_scored,
    "shuffled": shuffled,
    "argmax only": argmax_only,
}


def token_cap(frame: pd.DataFrame) -> pd.DataFrame:
    """Score 1 on the last token of an answer that hit the cap, 0 elsewhere."""
    out = frame.copy()
    scored = []
    for row in frame.itertuples(index=False):
        s = np.zeros(int(row.num_tokens), dtype=np.float32)
        if str(row.stop_reason) == "length" and s.size:
            s[-1] = 1.0
        scored.append(s)
    out["scores"] = scored
    return out


def best_accuracy(table: pd.DataFrame) -> float:
    """The highest accuracy any threshold on the response score could reach."""
    one = table.drop_duplicates(["prompt_id", "rollout_idx"])
    y = one["is_positive"].to_numpy(dtype=bool)
    s = one["rollout_score"].to_numpy(dtype=float)
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    positives, negatives = int(y.sum()), int((~y).sum())
    # predicting the top k as positive, for every k including none
    correct = np.concatenate([[negatives], tp + (negatives - fp)])
    return float(correct.max() / len(y))


def measure(
    val: pd.DataFrame, target: pd.DataFrame, budget: float, persistence: int
) -> Dict[str, float]:
    thresholds = choose_thresholds(val, [budget], persistence=persistence)
    table = per_rollout_table(target, thresholds, persistence=persistence)
    a = view_a_detection(table).iloc[0]
    b = view_b_coverage(target, thresholds).iloc[0]
    c = view_c_lead_time(table).iloc[0]
    # A scorer that cannot produce m flags in a row has every response score at
    # the floor of its range: the threshold lands on that floor, nothing fires
    # on validation, and on another split a floor a hair lower flips every
    # token to flagged. Neither reading says anything about the scorer, so
    # when the response scores take a single value the threshold-dependent
    # numbers are left blank and only the rank metrics, at chance, remain. A
    # threshold that merely cannot spend the budget because negatives tie at
    # the ceiling, as the token-cap rule's does, is still reported.
    saturated = bool(thresholds[0].saturated)
    one = table.drop_duplicates(["prompt_id", "rollout_idx"])
    degenerate = int(one["rollout_score"].nunique()) <= 1
    blank = float("nan")
    return {
        "tau": float(thresholds[0].tau),
        "saturated": saturated,
        "rollout_auc": rank_metrics(table)["rollout_auc"],
        "best_accuracy": best_accuracy(table),
        "degenerate": degenerate,
        "accuracy_at_budget": blank if degenerate else float(a["accuracy"]),
        "in_pattern_recall": blank if degenerate else float(b["in_pattern_recall"]),
        "warning_recall_256": blank if degenerate else float(b["warning_recall_256"]),
        "warning_recall_128": blank if degenerate else float(b["warning_recall_128"]),
        "preonset_coverage_256": blank if degenerate else float(b["preonset_coverage_256"]),
        "fired_before_frontier": blank if degenerate else float(c["fired_before_frontier"]),
        "median_offset": blank if degenerate else float(c["median_offset"]),
        "never_fired_positives": blank if degenerate else int(c["never_fired_positives"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer", nargs="+", action="append", metavar=("NAME", "DIR"), required=True,
                        help="a name followed by one scores directory per seed")
    parser.add_argument("--split", default=TUNING_SPLIT)
    parser.add_argument("--budget", type=float, default=0.01)
    parser.add_argument("--persistence", nargs="*", type=int, default=[1, 4, 8])
    parser.add_argument("--onset-labels", type=Path, default=None,
                        help="onset table naming the rollouts the judge could not rule on")
    parser.add_argument("--no-token-cap", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    unjudged = None
    if args.onset_labels is not None:
        from degeneration_probe.evaluation.scores import unjudged_rollouts
        unjudged = unjudged_rollouts(args.onset_labels)

    rows: List[dict] = []
    first_val = first_target = None
    for spec in args.scorer:
        name, dirs = spec[0], spec[1:]
        for seed_index, scores_dir in enumerate(dirs):
            scores_dir = Path(scores_dir)
            val = read_scores(scores_dir / "scores" / f"{TUNING_SPLIT}.parquet", split=TUNING_SPLIT, unjudged=unjudged)
            target = val if args.split == TUNING_SPLIT else read_scores(
                scores_dir / "scores" / f"{args.split}.parquet", split=args.split, unjudged=unjudged
            )
            if first_val is None:
                first_val, first_target = val, target
            for transform, fn in TRANSFORMS.items():
                rng_v, rng_t = np.random.default_rng(SEED + seed_index), np.random.default_rng(SEED + seed_index)
                tv = fn(val, rng_v)
                tt = tv if target is val else fn(target, rng_t)
                for m in args.persistence:
                    rows.append({"scorer": name, "seed_index": seed_index, "scores_dir": str(scores_dir),
                                 "transform": transform, "persistence": m, "split": args.split,
                                 **measure(tv, tt, args.budget, m)})
                    print(f"{name:<28} {transform:<12} m={m}: {rows[-1]['rollout_auc']:.4f} AUC, "
                          f"Warn256 {rows[-1]['warning_recall_256']:.4f}", flush=True)

    if not args.no_token_cap and first_val is not None:
        cap_val, cap_target = token_cap(first_val), token_cap(first_target)
        for transform, fn in TRANSFORMS.items():
            rng = np.random.default_rng(SEED)
            tv = fn(cap_val, rng)
            tt = tv if first_target is first_val else fn(cap_target, np.random.default_rng(SEED))
            for m in args.persistence:
                rows.append({"scorer": "token cap", "seed_index": 0, "scores_dir": "", "transform": transform,
                             "persistence": m, "split": args.split, **measure(tv, tt, args.budget, m)})

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    metrics = ["rollout_auc", "best_accuracy", "in_pattern_recall", "warning_recall_256", "preonset_coverage_256", "fired_before_frontier"]
    summary = table.groupby(["scorer", "transform", "persistence"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"))
    pd.set_option("display.width", 220)
    print(f"\n{len(table)} rows -> {args.out}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
