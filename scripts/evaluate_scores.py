"""Turn stored scores into the four views, and freeze the decision thresholds.

Pure CPU work over a table of per-token scores, so it can be re-run whenever
the protocol changes without recomputing a single score.

The boundary between tuning and reporting is enforced here rather than left to
discipline. Thresholds are chosen on validation data alone, written to
``decision_thresholds.json``, and read back unchanged for every other split. A
test split cannot be reported for a scorer that has no frozen thresholds.

    python scripts/evaluate_scores.py --run-dir outputs/<run_name>/<attempt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.protocol import (
    MIN_POSITIVES_FOR_REPORTING,
    Threshold,
    choose_thresholds,
    evaluate,
    per_rollout_table,
    view_c_lead_time,
    view_d_persistence,
)
from degeneration_probe.evaluation.scores import read_scores, unjudged_rollouts
from degeneration_probe.utils.file_utils import load_json, save_json

SCORES_DIR = "scores"
EVALUATION_DIR = "evaluation"
THRESHOLDS_FILE = "decision_thresholds.json"
TUNING_SPLIT = "val"


# A directory under one of these holds scores computed on a corpus other than the
# one the run was trained on, so the run's own configuration names the wrong build.
FOREIGN_CORPUS_MARKERS = ("cross_model", "lora_transplant")


def _discover_onset_labels(run_dir: Path) -> Optional[Path]:
    """The onset label table for the build a run was trained against.

    Returns nothing in the two cases where a guess would be wrong rather than
    merely absent. A scorer with no run configuration beside it, such as a
    model-free baseline, has nothing to read. And a directory of scores computed
    against another model's corpus sits below a run whose configuration names the
    training build, not the corpus the scores describe; which rollouts a judge
    could not rule on differs between corpora, so reading the wrong table drops the
    wrong rollouts. Both ask the caller to name the table instead.
    """
    parts = set(run_dir.parts)
    if parts & set(FOREIGN_CORPUS_MARKERS):
        return None
    for candidate in (run_dir, *run_dir.parents):
        config = candidate / "resolved_config.json"
        if config.is_file():
            build_root = load_json(config).get("dataset", {}).get("build_root")
            if build_root:
                return Path(build_root) / "onset_labels" / "onset_labels.parquet"
    return None


def population_shortfall(
    onset_labels: Path, split: str, unjudged: set, scored: int
) -> Optional[dict]:
    """How far a scored table falls short of the split it claims to cover.

    Scoring is meant to be exhaustive: no negative subsampled, no rollout capped,
    whatever the training configuration said, because a missing negative can only
    understate a false-alarm rate and the false-alarm rate is what every operating
    point here is solved from. That guarantee was documented and never checked, and
    a short table is invisible once it reaches the evaluator, which has nothing to
    compare it against. This supplies the comparison.
    """
    labels = pd.read_parquet(
        onset_labels, columns=["prompt_id", "rollout_idx", "split", "onset_resolution"]
    )
    in_split = labels[labels["split"] == split]
    expected = sum(
        (str(row.prompt_id), int(row.rollout_idx)) not in unjudged
        for row in in_split.itertuples(index=False)
    )
    if scored >= expected:
        return None
    return {"scored": int(scored), "corpus": int(expected), "missing": int(expected - scored)}


def _scores_path(run_dir: Path, split: str) -> Path:
    path = run_dir / SCORES_DIR / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No scores for split {split!r}: {path}. Run score_rollouts.py first.")
    return path


def freeze_thresholds(
    run_dir: Path,
    budgets: Sequence[float],
    persistence: int,
    unjudged: Optional[set] = None,
    shortfall: Optional[dict] = None,
) -> List[Threshold]:
    """Choose operating points on validation data and record them."""
    frame = read_scores(
        _scores_path(run_dir, TUNING_SPLIT), split=TUNING_SPLIT, unjudged=unjudged
    )
    thresholds = choose_thresholds(frame, budgets, persistence=persistence)
    payload = {
        "tuned_on": TUNING_SPLIT,
        "persistence": persistence,
        "population_shortfall": shortfall,
        "thresholds": [threshold.__dict__ for threshold in thresholds],
    }
    save_json(payload, run_dir / THRESHOLDS_FILE)
    return thresholds


def load_frozen_thresholds(run_dir: Path) -> tuple[List[Threshold], int]:
    path = run_dir / THRESHOLDS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No frozen thresholds at {path}. They are chosen on {TUNING_SPLIT!r} and must be "
            "frozen before any other split is reported."
        )
    payload = load_json(path)
    return [Threshold(**row) for row in payload["thresholds"]], int(payload["persistence"])


def compare_persistence(
    run_dir: Path,
    budgets: Sequence[float],
    candidates: Sequence[int],
    unjudged: Optional[set] = None,
) -> pd.DataFrame:
    """What each persistence window would cost and buy, on validation only.

    The run lengths of false alarms are what a persistence window should be
    chosen from: if they are a few tokens long while true alarms run for
    hundreds, a small window removes most spurious early stops for a few tokens
    of lead time. This puts that trade-off in one table instead of leaving it
    to be guessed.
    """
    frame = read_scores(
        _scores_path(run_dir, TUNING_SPLIT), split=TUNING_SPLIT, unjudged=unjudged
    )
    rows = []
    for persistence in candidates:
        thresholds = choose_thresholds(frame, budgets, persistence=persistence)
        table = per_rollout_table(frame, thresholds, persistence=persistence)
        lead = view_c_lead_time(table).set_index("target_negative_fpr")
        held = view_d_persistence(table)
        for threshold in thresholds:
            positive = held[
                (held["population"] == "positive")
                & (held["target_negative_fpr"] == threshold.target_negative_fpr)
            ]
            negative = held[
                (held["population"] == "negative")
                & (held["target_negative_fpr"] == threshold.target_negative_fpr)
            ]
            rows.append(
                {
                    "persistence": persistence,
                    "target_negative_fpr": threshold.target_negative_fpr,
                    "tau": threshold.tau,
                    "median_offset": lead.loc[threshold.target_negative_fpr, "median_offset"],
                    "never_fired_positives": lead.loc[
                        threshold.target_negative_fpr, "never_fired_positives"
                    ],
                    "false_early_stop_rate": lead.loc[
                        threshold.target_negative_fpr, "false_early_stop_rate"
                    ],
                    "positive_duty_cycle": float(positive["mean_duty_cycle"].iloc[0])
                    if len(positive)
                    else float("nan"),
                    "negative_median_run": float(negative["median_first_run_length"].iloc[0])
                    if len(negative)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def report_split(
    run_dir: Path,
    split: str,
    thresholds: Sequence[Threshold],
    persistence: int,
    min_positives: int,
    unjudged: Optional[set] = None,
) -> Path:
    frame = read_scores(_scores_path(run_dir, split), split=split, unjudged=unjudged)
    result = evaluate(frame, thresholds, persistence=persistence, min_positives=min_positives)
    out = run_dir / EVALUATION_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f"{name}.csv", index=False)
    save_json(
        {
            "split": split,
            "persistence": persistence,
            "thresholds": result["thresholds"],
            "rank_metrics": result["rank_metrics"],
        },
        out / "summary.json",
    )

    print(f"\n=== {split} ===")
    print("rank:", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["rank_metrics"].items()})
    for view in ("view_a_detection", "view_b_coverage", "view_c_lead_time", "view_d_persistence"):
        print(f"\n{view}")
        print(result[view].round(4).to_string(index=False))
    if split != TUNING_SPLIT:
        per_domain = result["per_domain_detection"]
        if len(per_domain):
            print("\nper_domain_detection (anecdotal cells are not estimates)")
            print(per_domain.round(4).to_string(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=None, help="Defaults to every scored split.")
    parser.add_argument(
        "--budgets",
        nargs="*",
        type=float,
        default=[0.01, 0.05, 0.10],
        help="Target false-alarm rates on validation negative rollouts.",
    )
    parser.add_argument("--persistence", type=int, default=1, help="The window to freeze.")
    parser.add_argument(
        "--compare-persistence",
        nargs="*",
        type=int,
        default=None,
        help="Also report what these windows would buy, on validation only.",
    )
    parser.add_argument("--min-positives", type=int, default=MIN_POSITIVES_FOR_REPORTING)
    parser.add_argument(
        "--reuse-thresholds",
        action="store_true",
        help="Report against the frozen thresholds without choosing them again.",
    )
    parser.add_argument(
        "--onset-labels",
        type=Path,
        default=None,
        help="The onset label table. Rollouts the judge could not rule on are left "
        "out of the population. Discovered from the run's resolved_config.json when "
        "not given.",
    )
    parser.add_argument(
        "--keep-unjudged",
        action="store_true",
        help="Count capped answers with no resolved onset among the healthy ones. "
        "Only for reproducing a number measured before they were excluded.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    unjudged = set()
    if not args.keep_unjudged:
        labels_path = args.onset_labels or _discover_onset_labels(run_dir)
        if labels_path is None:
            foreign = set(run_dir.parts) & set(FOREIGN_CORPUS_MARKERS)
            reason = (
                f"these scores were computed against another model's corpus, so the "
                f"build named in the run's own configuration is the wrong one"
                if foreign
                else f"{run_dir} has no resolved_config.json to read a build root from"
            )
            raise SystemExit(
                f"Cannot tell which rollouts the judge could not rule on: {reason}. "
                "Pass --onset-labels naming the table for the corpus these scores "
                "describe, or --keep-unjudged to count them as healthy and say so."
            )
        unjudged = unjudged_rollouts(labels_path)
        print(f"Excluding {len(unjudged)} rollouts with no judged outcome ({labels_path})")

    if args.compare_persistence:
        comparison = compare_persistence(
            run_dir, args.budgets, args.compare_persistence, unjudged=unjudged
        )
        (run_dir / EVALUATION_DIR).mkdir(parents=True, exist_ok=True)
        comparison.to_csv(run_dir / EVALUATION_DIR / "persistence_comparison.csv", index=False)
        print("Persistence trade-off, on validation only:")
        print(comparison.round(4).to_string(index=False))

    shortfall = None
    if not args.keep_unjudged and labels_path is not None:
        scored = len(read_scores(_scores_path(run_dir, TUNING_SPLIT), split=TUNING_SPLIT,
                                unjudged=unjudged, validate=False))
        shortfall = population_shortfall(labels_path, TUNING_SPLIT, unjudged, scored)
        if shortfall:
            print(
                f"\n  WARNING: this table covers {shortfall['scored']} of the "
                f"{shortfall['corpus']} rollouts in {TUNING_SPLIT}, so "
                f"{shortfall['missing']} are missing. Every threshold below is solved "
                "over a population that is short, and a missing negative can only "
                "understate a false-alarm rate. Re-score the split before reporting "
                "from it.\n"
            )

    if args.reuse_thresholds:
        thresholds, persistence = load_frozen_thresholds(run_dir)
    else:
        thresholds = freeze_thresholds(
            run_dir, args.budgets, args.persistence, unjudged=unjudged,
            shortfall=shortfall,
        )
        persistence = args.persistence
        print(f"\nFrozen on {TUNING_SPLIT} (m={persistence}) -> {run_dir / THRESHOLDS_FILE}")
        for threshold in thresholds:
            print(
                f"  budget {threshold.target_negative_fpr:>5.0%}  tau={threshold.tau:.6f}  "
                f"realized={threshold.realized_negative_fpr:.4f}  "
                f"over {threshold.negative_rollouts} negative rollouts"
            )
        if any(threshold.saturated for threshold in thresholds):
            worst = max(t.tied_at_ceiling for t in thresholds if t.saturated)
            print(
                f"\n  WARNING: {worst:.1%} of negative rollouts tie at this scorer's highest "
                "score.\n  No threshold can spend a budget smaller than that, so the "
                "thresholds above\n  sit past the top of the range and nothing fires. The "
                "empty columns that\n  follow mean the scorer cannot separate these rollouts, "
                "not that it stayed\n  quiet on them."
            )

    splits = args.splits
    if splits is None:
        splits = sorted(path.stem for path in (run_dir / SCORES_DIR).glob("*.parquet"))
    for split in splits:
        report_split(
            run_dir, split, thresholds, persistence, args.min_positives, unjudged=unjudged
        )


if __name__ == "__main__":
    main()
