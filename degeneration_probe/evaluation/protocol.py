"""One evaluation protocol, applied to any per-token scorer.

Everything here is built on the first-alarm position: for a rollout r, a
decision threshold tau and a persistence window m,

    a_r(tau, m) = min { t : p_r(t') >= tau for all t' in [t, t+m) }

or infinity when the scorer never fires. This is the first token at which the
scorer would have acted, requiring m consecutive tokens at or above the
threshold before committing. From it, four views are reported:

    A  rollout-level detection: did the scorer catch this rollout at all
    B  token-level coverage: false positives on healthy text, recall inside
       the degenerate pattern
    C  lead time: how far before or after the frontier the alarm landed
    D  alarm persistence: having fired, does the scorer stay convinced

The four are reported together at a fixed threshold, because none of them is
interpretable alone. A scorer that fires on every token has perfect recall and
perfect persistence and is useless; only its false-positive rate says so.

Sweeping tau is free here. The alarm position is monotone in tau, so one pass
over a rollout yields the whole threshold-to-alarm mapping as a staircase, and
every threshold afterwards is a lookup rather than another pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Domains backed by fewer positives than this are reported but never quoted as
# an estimate of anything: at that count a single rollout moves every rate.
MIN_POSITIVES_FOR_REPORTING = 10


# --- the first-alarm machinery -------------------------------------------------


def persistence_scores(scores: np.ndarray, persistence: int) -> np.ndarray:
    """The score a firing rule actually sees, given it needs m tokens in a row.

    Position t carries the minimum over [t, t+m), so a single spike can no
    longer trip the rule while a sustained run still does.
    """
    if persistence < 1:
        raise ValueError(f"persistence must be at least 1, got {persistence}")
    scores = np.asarray(scores, dtype=np.float64)
    if persistence == 1:
        return scores
    if scores.size < persistence:
        return np.empty(0, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(scores, persistence)
    return windows.min(axis=1)


def alarm_staircase(scores: np.ndarray, persistence: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Every threshold's alarm position, as (positions, thresholds_reached).

    Raising the threshold can only delay an alarm, so the running maximum of
    the persistence scores is a staircase: at each step the position is the
    first token that reaches that level. One pass answers every threshold.
    """
    effective = persistence_scores(scores, persistence)
    if effective.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    running = np.maximum.accumulate(effective)
    rises = np.flatnonzero(np.r_[True, running[1:] > running[:-1]])
    return rises.astype(np.int64), running[rises]


def first_alarm(scores: np.ndarray, threshold: float, persistence: int = 1) -> Optional[int]:
    """The first token at which the scorer would have fired, or None."""
    positions, levels = alarm_staircase(scores, persistence)
    if positions.size == 0:
        return None
    index = int(np.searchsorted(levels, threshold, side="left"))
    if index >= levels.size:
        return None
    return int(positions[index])


def rollout_score(scores: np.ndarray, persistence: int = 1) -> float:
    """The threshold-free ranking score: the highest level ever reached.

    A rollout fires at tau exactly when this is at least tau, so the ranking
    used for a threshold-free area and the decision made at a fixed threshold
    describe the same quantity.
    """
    effective = persistence_scores(scores, persistence)
    return float(effective.max()) if effective.size else 0.0


# --- choosing and freezing thresholds ------------------------------------------


@dataclass(frozen=True)
class Threshold:
    """A decision threshold, and the false-alarm budget it was chosen for."""

    target_negative_fpr: float
    tau: float
    realized_negative_fpr: float
    negative_rollouts: int


def threshold_for_budget(negative_scores: np.ndarray, budget: float) -> Tuple[float, float]:
    """The lowest threshold whose false-alarm rate stays inside the budget."""
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"a false-alarm budget must lie in [0, 1], got {budget}")
    scores = np.sort(np.asarray(negative_scores, dtype=np.float64))[::-1]
    if scores.size == 0:
        raise ValueError("choosing a threshold needs negative rollouts to measure against")
    allowed = int(np.floor(budget * scores.size))
    if allowed >= scores.size:
        tau = 0.0
    else:
        # Sit just above the highest score that must not fire. Stepping toward
        # infinity rather than toward one matters at the top of the range: a
        # zero budget against a scorer that reaches 1.0 needs a threshold above
        # 1.0, and stepping toward 1.0 from 1.0 would not move at all.
        tau = float(np.nextafter(scores[allowed], np.inf))
    realized = float((scores >= tau).mean())
    return tau, realized


def choose_thresholds(
    frame: pd.DataFrame,
    budgets: Sequence[float],
    *,
    persistence: int = 1,
    tuning_split: str = "val",
) -> List[Threshold]:
    """Freeze a family of operating points, on validation data only.

    A single threshold would bake in a deployment decision this protocol cannot
    settle, namely how much false-alarm cost is acceptable, so a small family is
    frozen instead and every view is reported at each of them.
    """
    splits = set(frame["split"].unique())
    if splits != {tuning_split}:
        raise ValueError(
            f"thresholds may only be chosen on {tuning_split!r}, got splits {sorted(splits)}; "
            "choosing them anywhere else would tune on the data being reported"
        )
    negatives = frame[~frame["is_positive"].astype(bool)]
    if negatives.empty:
        raise ValueError("no negative rollouts to measure a false-alarm rate against")
    scores = np.array(
        [rollout_score(row, persistence) for row in negatives["scores"]], dtype=np.float64
    )
    thresholds = []
    for budget in budgets:
        tau, realized = threshold_for_budget(scores, budget)
        thresholds.append(
            Threshold(
                target_negative_fpr=float(budget),
                tau=tau,
                realized_negative_fpr=realized,
                negative_rollouts=int(scores.size),
            )
        )
    return thresholds


# --- per-rollout measurements --------------------------------------------------


def _persistence_metrics(above: np.ndarray, alarm: Optional[int]) -> Dict[str, float]:
    """How the scorer behaves after it first fires, and how often it flips."""
    total = above.size
    transitions = {
        "fire_to_fire": int(np.count_nonzero(above[:-1] & above[1:])) if total > 1 else 0,
        "fire_total": int(np.count_nonzero(above[:-1])) if total > 1 else 0,
        "quiet_to_fire": int(np.count_nonzero(~above[:-1] & above[1:])) if total > 1 else 0,
        "quiet_total": int(np.count_nonzero(~above[:-1])) if total > 1 else 0,
    }
    changes = np.diff(above.astype(np.int8))
    episodes = int(above[0]) + int(np.count_nonzero(changes == 1)) if total else 0

    metrics: Dict[str, float] = {
        "episodes": episodes,
        "tokens_above": int(np.count_nonzero(above)),
        **transitions,
    }
    if alarm is None:
        metrics.update(
            first_run_length=np.nan,
            first_run_fraction=np.nan,
            duty_cycle=np.nan,
            retracted=np.nan,
            tokens_to_commitment=np.nan,
        )
        return metrics

    tail = above[alarm:]
    if tail.all():
        first_run = int(tail.size)
    else:
        first_run = int(np.argmin(tail))
    # Retraction needs the first run to have actually ended: a run that reaches
    # the last token leaves nothing behind it, which is holding, not retracting.
    run_end = alarm + first_run
    silent_after = run_end < above.size and not bool(above[run_end:].any())
    if above.size and above[-1]:
        quiet = np.flatnonzero(~above)
        commitment = int(quiet[-1]) + 1 if quiet.size else 0
        tokens_to_commitment = float(max(commitment - alarm, 0))
    else:
        tokens_to_commitment = np.inf

    metrics.update(
        first_run_length=float(first_run),
        first_run_fraction=float(first_run / tail.size) if tail.size else np.nan,
        duty_cycle=float(tail.mean()),
        retracted=float(silent_after),
        tokens_to_commitment=tokens_to_commitment,
    )
    return metrics


def per_rollout_table(
    frame: pd.DataFrame, thresholds: Sequence[Threshold], *, persistence: int = 1
) -> pd.DataFrame:
    """One row per rollout per threshold: the raw material for every view.

    Kept tidy rather than pre-aggregated, so a later question that nobody has
    asked yet is a group-by rather than another pass over the scores.
    """
    rows = []
    for row in frame.itertuples(index=False):
        scores = np.asarray(row.scores, dtype=np.float64)
        positions, levels = alarm_staircase(scores, persistence)
        peak = float(levels[-1]) if levels.size else 0.0
        onset = int(row.onset_position) if row.is_positive else None
        for threshold in thresholds:
            index = int(np.searchsorted(levels, threshold.tau, side="left"))
            alarm = int(positions[index]) if index < levels.size else None
            above = scores >= threshold.tau
            record = {
                "prompt_id": row.prompt_id,
                "rollout_idx": row.rollout_idx,
                "domain": row.domain,
                "split": row.split,
                "is_positive": bool(row.is_positive),
                "num_tokens": int(row.num_tokens),
                "onset_position": onset,
                "target_negative_fpr": threshold.target_negative_fpr,
                "tau": threshold.tau,
                "rollout_score": peak,
                "alarm": alarm,
                "fired": alarm is not None,
                "offset": float(alarm - onset) if (alarm is not None and onset is not None) else np.nan,
            }
            record.update(_persistence_metrics(above, alarm))
            if onset is not None:
                in_pattern = above[onset:]
                record["in_pattern_tokens"] = int(in_pattern.size)
                record["in_pattern_hits"] = int(np.count_nonzero(in_pattern))
                record["in_pattern_misses"] = int(in_pattern.size - np.count_nonzero(in_pattern))
            else:
                record["in_pattern_tokens"] = 0
                record["in_pattern_hits"] = 0
                record["in_pattern_misses"] = 0
            rows.append(record)
    return pd.DataFrame(rows)


# --- the four views ------------------------------------------------------------


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def view_a_detection(table: pd.DataFrame) -> pd.DataFrame:
    """Rollout-level detection: the confusion matrix at each threshold."""
    rows = []
    for tau_target, group in table.groupby("target_negative_fpr", sort=True):
        positives = group["is_positive"].to_numpy(dtype=bool)
        fired = group["fired"].to_numpy(dtype=bool)
        true_positive = int(np.count_nonzero(positives & fired))
        false_positive = int(np.count_nonzero(~positives & fired))
        false_negative = int(np.count_nonzero(positives & ~fired))
        true_negative = int(np.count_nonzero(~positives & ~fired))
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        rows.append(
            {
                "target_negative_fpr": tau_target,
                "tau": float(group["tau"].iloc[0]),
                "positives": int(positives.sum()),
                "negatives": int((~positives).sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
                "f1": _safe_ratio(2 * precision * recall, precision + recall)
                if precision and recall
                else np.nan,
                "accuracy": _safe_ratio(true_positive + true_negative, len(group)),
                "negative_fpr": _safe_ratio(false_positive, false_positive + true_negative),
            }
        )
    return pd.DataFrame(rows)


def rank_metrics(table: pd.DataFrame) -> Dict[str, float]:
    """Threshold-free summary, as a check that no operating point was special."""
    one_threshold = table.drop_duplicates(["prompt_id", "rollout_idx"])
    labels = one_threshold["is_positive"].to_numpy(dtype=bool)
    scores = one_threshold["rollout_score"].to_numpy(dtype=float)
    if labels.all() or not labels.any():
        return {"rollout_auc": np.nan, "rollout_ap": np.nan, "positives": int(labels.sum())}
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "rollout_auc": float(roc_auc_score(labels, scores)),
        "rollout_ap": float(average_precision_score(labels, scores)),
        "positives": int(labels.sum()),
    }


def view_b_coverage(frame: pd.DataFrame, thresholds: Sequence[Threshold]) -> pd.DataFrame:
    """Token-level coverage, reported per population and never blended.

    The in-pattern tail of a positive rollout dominates the positive token
    population and is trivially separable, so one blended token accuracy would
    read close to perfect for almost any scorer. Rates therefore stay split by
    population, and every one is reported next to the size of the population it
    came from.
    """
    rows = []
    for threshold in thresholds:
        negative_tokens = negative_hits = 0
        negative_rollouts = negative_rollouts_with_hit = 0
        in_pattern_tokens = in_pattern_hits = 0
        pre_frontier_tokens = 0
        positive_rollouts = 0
        for row in frame.itertuples(index=False):
            scores = np.asarray(row.scores, dtype=np.float64)
            above = scores >= threshold.tau
            if row.is_positive:
                onset = int(row.onset_position)
                positive_rollouts += 1
                pre_frontier_tokens += onset
                in_pattern_tokens += int(above[onset:].size)
                in_pattern_hits += int(np.count_nonzero(above[onset:]))
            else:
                negative_rollouts += 1
                negative_tokens += int(above.size)
                hits = int(np.count_nonzero(above))
                negative_hits += hits
                negative_rollouts_with_hit += int(hits > 0)
        rows.append(
            {
                "target_negative_fpr": threshold.target_negative_fpr,
                "tau": threshold.tau,
                "negative_rollouts": negative_rollouts,
                "negative_tokens": negative_tokens,
                "token_false_positive_rate": _safe_ratio(negative_hits, negative_tokens),
                "negative_rollouts_with_any_false_positive": _safe_ratio(
                    negative_rollouts_with_hit, negative_rollouts
                ),
                "positive_rollouts": positive_rollouts,
                "in_pattern_tokens": in_pattern_tokens,
                "in_pattern_recall": _safe_ratio(in_pattern_hits, in_pattern_tokens),
                "pre_frontier_tokens": pre_frontier_tokens,
            }
        )
    return pd.DataFrame(rows)


def view_c_lead_time(table: pd.DataFrame) -> pd.DataFrame:
    """How far before or after the frontier the alarm landed.

    Negative offsets are lead time gained. Positives that never fired are
    counted separately rather than folded into an average, where they would
    silently flatter it.
    """
    rows = []
    for tau_target, group in table.groupby("target_negative_fpr", sort=True):
        positives = group[group["is_positive"]]
        detected = positives[positives["fired"]]
        offsets = detected["offset"].to_numpy(dtype=float)
        negatives = group[~group["is_positive"]]
        rows.append(
            {
                "target_negative_fpr": tau_target,
                "tau": float(group["tau"].iloc[0]),
                "detected_positives": int(len(detected)),
                "never_fired_positives": int(len(positives) - len(detected)),
                "median_offset": float(np.median(offsets)) if offsets.size else np.nan,
                "mean_offset": float(offsets.mean()) if offsets.size else np.nan,
                "median_absolute_offset": float(np.median(np.abs(offsets)))
                if offsets.size
                else np.nan,
                "fired_before_frontier": _safe_ratio(int((offsets < 0).sum()), offsets.size),
                "false_early_stop_rate": _safe_ratio(
                    int(negatives["fired"].sum()), len(negatives)
                ),
            }
        )
    return pd.DataFrame(rows)


def view_d_persistence(table: pd.DataFrame) -> pd.DataFrame:
    """Having fired, does the scorer stay convinced.

    Read in opposite directions for the two populations. On positives a sticky
    alarm is the desired behaviour. On negatives every alarm is an error, and
    the run lengths separate a jittery scorer, whose false alarms are spikes a
    few tokens long, from a confidently wrong one that fires and holds. Those
    two failures call for different fixes, and the run lengths here are what
    the persistence window m should be chosen from.
    """
    rows = []
    for (tau_target, is_positive), group in table.groupby(
        ["target_negative_fpr", "is_positive"], sort=True
    ):
        fired = group[group["fired"]]
        commitment = fired["tokens_to_commitment"].replace([np.inf], np.nan).dropna()
        rows.append(
            {
                "target_negative_fpr": tau_target,
                "tau": float(group["tau"].iloc[0]),
                "population": "positive" if is_positive else "negative",
                "rollouts": int(len(group)),
                "fired": int(len(fired)),
                "median_first_run_length": float(fired["first_run_length"].median())
                if len(fired)
                else np.nan,
                "median_first_run_fraction": float(fired["first_run_fraction"].median())
                if len(fired)
                else np.nan,
                "mean_duty_cycle": float(fired["duty_cycle"].mean()) if len(fired) else np.nan,
                "median_episodes": float(fired["episodes"].median()) if len(fired) else np.nan,
                "retraction_rate": float(fired["retracted"].mean()) if len(fired) else np.nan,
                "median_tokens_to_commitment": float(commitment.median())
                if len(commitment)
                else np.nan,
                "never_commits": int(len(fired) - len(commitment)),
                "stickiness": _safe_ratio(
                    group["fire_to_fire"].sum(), group["fire_total"].sum()
                ),
                "spontaneous_firing_rate": _safe_ratio(
                    group["quiet_to_fire"].sum(), group["quiet_total"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def per_domain_detection(
    table: pd.DataFrame, *, min_positives: int = MIN_POSITIVES_FOR_REPORTING
) -> pd.DataFrame:
    """View A per domain, with underpowered cells marked rather than hidden.

    Held-out domains differ sharply in how often they degenerate, and at least
    one yields a handful of positives in total, so pooling them would hide the
    fact that a precision came from a single rollout.
    """
    frames = []
    for domain, group in table.groupby("domain", sort=True):
        rows = view_a_detection(group)
        rows.insert(0, "domain", domain)
        rows["anecdotal"] = rows["positives"] < min_positives
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate(
    frame: pd.DataFrame,
    thresholds: Sequence[Threshold],
    *,
    persistence: int = 1,
    min_positives: int = MIN_POSITIVES_FOR_REPORTING,
) -> Dict[str, object]:
    """Run the whole protocol over one scored split."""
    table = per_rollout_table(frame, thresholds, persistence=persistence)
    return {
        "persistence": persistence,
        "thresholds": [threshold.__dict__ for threshold in thresholds],
        "rank_metrics": rank_metrics(table),
        "view_a_detection": view_a_detection(table),
        "view_b_coverage": view_b_coverage(frame, thresholds),
        "view_c_lead_time": view_c_lead_time(table),
        "view_d_persistence": view_d_persistence(table),
        "per_domain_detection": per_domain_detection(table, min_positives=min_positives),
        "rollout_table": table,
    }
