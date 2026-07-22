"""Shared evaluation logic for the three structural degeneration metrics
(Entropy, TTR, LRS) against the LLM-judge ground truth.

See docs/analysis/metric_evaluation_unification_plan.md for the full design
this implements. In short: one population, shared across all three metrics --

  - **Positives**: rollouts in the LLM-judge's truncated-population sample
    (`llm_judge_sample_path`) with a successful, judge-confirmed
    `is_degenerating == True` verdict.
  - **Negatives**: every `stop_reason == "eos"` rollout, by definition (no
    judge call needed), plus the small number of judge-confirmed-negative
    truncated rollouts (kept as hard negatives).
  - A deterministic prompt-hash split divides the population into
    "calibration" (threshold tuning) and "test" (everything reported), so no
    metric is tuned and reported on the same rows.

Each metric's own per-rollout score (TTR's `max_repetition_score`, LRS's
`onset_period_repeat_count`, Entropy's drop score) is supplied by the
calling notebook -- this module only knows about the population, the split,
confusion metrics, and aggregation, never how a score was computed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

# Domains with fewer than this many *test-side* positive rollouts are
# reported (never silently hidden) but excluded from any aggregate --
# see docs/analysis/metric_evaluation_unification_plan.md, open question #1.
MIN_POSITIVES_TEST = 10

# Kept unchanged from the original entropy-only notebook: renaming this tag
# would just reassign which prompts land in which bucket for no benefit, and
# the split is already validated (disjoint calibration/test prompt sets).
SPLIT_HASH_TAG = "entropy-inspection-v1"

# The only backend with real (non-failed) judge verdicts as of 2026-07-22 --
# `results_anthropic.parquet` is entirely status=="failed" (Anthropic
# billing error), so it must never silently win a merge against this one.
JUDGE_BACKEND = "claude_agent_sdk"


def prompt_split(prompt_id: str) -> str:
    """Deterministic 50/50 calibration/test split, keyed only by prompt_id
    so every rollout of a prompt lands in the same bucket (no leakage)."""
    digest = hashlib.sha256(f"{SPLIT_HASH_TAG}:{prompt_id}".encode()).digest()
    return "test" if int.from_bytes(digest[:8], "big") < 2**63 else "calibration"


def build_unified_population(
    rollout_signal_df: pd.DataFrame,
    llm_judge_sample_df: pd.DataFrame,
    llm_judge_results_df: pd.DataFrame,
    backend: Optional[str] = JUDGE_BACKEND,
) -> pd.DataFrame:
    """One row per usable rollout: domain, prompt_id, rollout_idx,
    is_positive, onset_quote, evaluation_split, population ("truncated_judged"
    or "eos_negative"). Truncated rollouts with no successful judge verdict
    (failed or not yet judged) are dropped -- there is no usable ground truth
    for them either way, not silently treated as negative.

    `backend` filters `llm_judge_results_df` to one judge backend when the
    frame has a `"backend"` column (e.g. inspect_dataset.ipynb, which
    concatenates every backend's results file into one frame). Pass
    `backend=None`, or a frame with no `"backend"` column, when the caller
    already loaded a single backend's results file directly (e.g.
    ttr_inspection.ipynb) -- the filter is then skipped rather than raising.
    """
    results = llm_judge_results_df
    if backend is not None and "backend" in results.columns:
        results = results[results["backend"] == backend]
    judge_ok = results[results["status"] == "ok"][
        ["prompt_id", "rollout_idx", "is_degenerating", "onset_quote"]
    ]

    truncated = llm_judge_sample_df.merge(judge_ok, on=["prompt_id", "rollout_idx"], how="left")
    truncated = truncated.dropna(subset=["is_degenerating"]).copy()
    truncated["is_positive"] = truncated["is_degenerating"].astype(bool)
    truncated["population"] = "truncated_judged"
    truncated = truncated[["domain", "prompt_id", "rollout_idx", "is_positive", "onset_quote", "population"]]

    eos = rollout_signal_df.loc[
        rollout_signal_df["stop_reason"] == "eos", ["domain", "prompt_id", "rollout_idx"]
    ].copy()
    eos["is_positive"] = False
    eos["onset_quote"] = None
    eos["population"] = "eos_negative"

    population = pd.concat([truncated, eos], ignore_index=True)
    population["evaluation_split"] = population["prompt_id"].apply(prompt_split)
    return population


def confusion_metrics(is_positive: pd.Series, predicted: pd.Series) -> dict:
    """The shared confusion-metric set reported for every metric: tp/fp/fn/tn
    counts plus precision/recall/specificity/balanced_accuracy/mcc/accuracy."""
    is_positive = is_positive.astype(bool)
    predicted = predicted.astype(bool)
    tp = int((is_positive & predicted).sum())
    fp = int((~is_positive & predicted).sum())
    fn = int((is_positive & ~predicted).sum())
    tn = int((~is_positive & ~predicted).sum())
    n = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_accuracy = np.nanmean([recall, specificity])
    mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")

    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": balanced_accuracy, "mcc": mcc, "accuracy": accuracy,
    }


def tune_threshold(
    score: pd.Series, is_positive: pd.Series, candidates: Sequence[float], higher_is_positive: bool = True
) -> float:
    """Pick the candidate threshold maximizing balanced accuracy on the given
    (calibration-only) rows. Ties select the higher threshold, matching the
    tie-break already used for entropy's tau."""
    best_threshold, best_balanced_accuracy = None, -1.0
    for threshold in sorted(candidates):
        predicted = score >= threshold if higher_is_positive else score <= threshold
        metrics = confusion_metrics(is_positive, predicted)
        balanced_accuracy = metrics["balanced_accuracy"]
        if not np.isnan(balanced_accuracy) and balanced_accuracy >= best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold
    if best_threshold is None:
        raise ValueError("No candidate threshold produced a defined balanced accuracy")
    return best_threshold


def per_domain_table(
    is_positive: pd.Series, predicted: pd.Series, domain: pd.Series, min_positives_test: int = MIN_POSITIVES_TEST
) -> pd.DataFrame:
    """Per-domain confusion metrics (rows to report as-is, on the test
    split), each flagged `insufficient` when its positive count is below
    `min_positives_test` -- reported, never hidden, but excluded from
    `macro_average_row` below."""
    rows = []
    for domain_name, mask in pd.Series(domain).groupby(domain).groups.items():
        metrics = confusion_metrics(is_positive.loc[mask], predicted.loc[mask])
        metrics["domain"] = domain_name
        metrics["insufficient"] = metrics["tp"] + metrics["fn"] < min_positives_test
        rows.append(metrics)
    table = pd.DataFrame(rows).set_index("domain")
    column_order = ["n", "tp", "fp", "fn", "tn", "precision", "recall", "specificity",
                     "balanced_accuracy", "mcc", "accuracy", "insufficient"]
    return table[column_order]


def macro_average_row(per_domain: pd.DataFrame) -> pd.Series:
    """Domain-balanced aggregate: equal-weight mean of each metric across
    domains NOT flagged insufficient (never row-subsampling -- see plan)."""
    usable = per_domain[~per_domain["insufficient"]]
    metric_cols = ["precision", "recall", "specificity", "balanced_accuracy", "mcc", "accuracy"]
    row = usable[metric_cols].mean()
    row["n_domains"] = len(usable)
    row["n"] = int(usable["n"].sum())
    return row


def pooled_row(is_positive: pd.Series, predicted: pd.Series) -> dict:
    """Simple pooled confusion metrics across all rows (no domain
    weighting) -- kept alongside the macro-average for direct comparison,
    since it's what the pre-unification tables reported."""
    return confusion_metrics(is_positive, predicted)


def roc_pr_curve(score: pd.Series, is_positive: pd.Series, higher_is_positive: bool = True) -> dict:
    """Threshold-free ROC/PR curves + AUC/average-precision, computed on
    whichever rows are passed in (intended: the test split), as a sanity
    check that the single calibrated operating point wasn't a lucky pick.
    `score` is oriented so that higher always means "more likely positive"
    before handing off to sklearn. Rows with an undefined score (e.g. a
    rollout too short for the metric to produce one) are dropped -- they
    cannot be ranked, not silently treated as a low or high score.
    """
    defined = score.notna()
    y = is_positive[defined].astype(int).to_numpy()
    s = score[defined].to_numpy(dtype=float)
    if not higher_is_positive:
        s = -s
    fpr, tpr, roc_thresholds = roc_curve(y, s)
    precision, recall, pr_thresholds = precision_recall_curve(y, s)
    return {
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "roc_auc": roc_auc_score(y, s),
        "pr": pd.DataFrame({"precision": precision, "recall": recall}),
        "average_precision": average_precision_score(y, s),
    }


@dataclass
class OnsetOffsetStats:
    n_true_positive: int
    n_fired: int
    n_never_fired: int
    median_signed_offset: float
    median_abs_offset: float
    mean_signed_offset: float
    mean_abs_offset: float
    false_early_stop_rate: float


def onset_offset_stats(
    true_positive_mask: pd.Series,
    first_alarm_position: pd.Series,
    judge_onset_position: pd.Series,
    true_negative_mask: pd.Series,
    fired_on_negative_mask: pd.Series,
) -> OnsetOffsetStats:
    """View C: for true positives, how far the metric's own first-alarm
    position sits from the judge's onset position (negative = the metric
    fires earlier than the judge -- lead time gained); for true negatives,
    how often the metric fires at all (the direct cost of using it as an
    early-stopping trigger). `first_alarm_position` is NaN/None for
    rollouts where the metric never fires.
    """
    fired = true_positive_mask & first_alarm_position.notna()
    offset = (first_alarm_position[fired] - judge_onset_position[fired]).astype(float)

    n_true_positive = int(true_positive_mask.sum())
    n_fired = int(fired.sum())
    n_true_negative = int(true_negative_mask.sum())
    n_fired_on_negative = int((true_negative_mask & fired_on_negative_mask).sum())

    return OnsetOffsetStats(
        n_true_positive=n_true_positive,
        n_fired=n_fired,
        n_never_fired=n_true_positive - n_fired,
        median_signed_offset=float(offset.median()) if n_fired else float("nan"),
        median_abs_offset=float(offset.abs().median()) if n_fired else float("nan"),
        mean_signed_offset=float(offset.mean()) if n_fired else float("nan"),
        mean_abs_offset=float(offset.abs().mean()) if n_fired else float("nan"),
        false_early_stop_rate=n_fired_on_negative / n_true_negative if n_true_negative else float("nan"),
    )
