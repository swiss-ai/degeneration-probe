"""Deciding, per depth, which checkpoint to keep and when to stop training it.

A run trains a head at every depth and saves all of them every fifty steps.
Something has to say which of those saved steps is the one a depth is judged on,
and when a depth has stopped improving enough to be worth training further.

The quantity that decision runs on is coverage of the tokens immediately before
the frontier. Two properties make it the right one and both matter:

- It is a mean over tens of thousands of tokens, so it moves smoothly between
  neighbouring checkpoints, where a median over a hundred first-alarm positions
  jumps around by more than it moves over a whole run.
- It is defined over every degenerate rollout. A rollout that is never flagged
  lowers it, rather than dropping out of it the way it drops out of a median
  taken over the rollouts that fired. A scorer cannot improve its standing by
  missing the hard cases.

Earliness is what the project is about, so it is what selection runs on. But a
head that has not yet learned to flag the loop itself can post a flattering
number before the loop by accident, so a head only becomes selectable once its
coverage inside the loop clears a floor. That floor governs what is kept, never
whether training continues: a head that plateaus below it has stopped learning
and should stop, and because a run ends only when its last depth stops, a head
that could never stop would hold the whole run open.

Everything here is a pure function of recorded numbers, so the rule can be
applied while a run trains or replayed afterwards over saved checkpoints, and
the two give the same answer by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from degeneration_probe.evaluation.protocol import (
    WARNING_BANDS,
    coverage_window,
    threshold_for_budget,
)

# The false-alarm rate the rule steers on, measured per rollout rather than per
# token, because a rollout is what a deployment would act on.
STEERING_BUDGET = 0.01


# --- the record one depth produces at one step ---------------------------------


def validation_record(
    *,
    negative_peaks: Sequence[float],
    positive_scores: Sequence[np.ndarray],
    onsets: Sequence[int],
    budget: float = STEERING_BUDGET,
    bands: Sequence[int] = WARNING_BANDS,
) -> Dict[str, float]:
    """Everything one depth reports at one step, from its scores.

    ``negative_peaks`` is the highest score each healthy rollout reached, which
    is all that a rollout-level threshold needs from them. ``positive_scores``
    holds every token of every degenerate rollout, because coverage is a
    statement about tokens and the threshold is not known until the healthy
    rollouts have been read.

    This is the single definition. Training calls it with scores from the live
    forward pass and a replay calls it with scores from applying saved weights
    to cached activations; neither has a copy of its own.
    """
    negatives = np.asarray(negative_peaks, dtype=np.float64)
    if negatives.size == 0 or len(positive_scores) == 0:
        return {}
    tau, realized = threshold_for_budget(negatives, budget)

    peaks = np.array([float(np.max(s)) if len(s) else 0.0 for s in positive_scores])
    inside_hits = inside_total = 0
    band_hits = {band: 0 for band in bands}
    band_total = {band: 0 for band in bands}
    offsets: List[float] = []
    never_fired = 0

    for scores, onset in zip(positive_scores, onsets):
        scores = np.asarray(scores, dtype=np.float64)
        above = scores >= tau
        onset = int(onset)

        inside = above[coverage_window(scores.size, onset, None)]
        inside_hits += int(np.count_nonzero(inside))
        inside_total += int(inside.size)
        for band in bands:
            run_up = above[coverage_window(scores.size, onset, band)]
            band_hits[band] += int(np.count_nonzero(run_up))
            band_total[band] += int(run_up.size)

        fired = np.flatnonzero(above)
        if fired.size:
            offsets.append(float(fired[0] - onset))
        else:
            never_fired += 1

    record = {
        "budget_tau": float(tau),
        "budget_realized_fpr": float(realized),
        "recall_at_budget": float(np.count_nonzero(peaks >= tau) / peaks.size),
        "in_pattern_recall": _ratio(inside_hits, inside_total),
        "never_fired_positives": float(never_fired),
        "median_offset": float(np.median(offsets)) if offsets else float("nan"),
    }
    for band in bands:
        record[f"warning_recall_{band}"] = _ratio(band_hits[band], band_total[band])
    return record


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


# --- the stopping rule ---------------------------------------------------------


@dataclass(frozen=True)
class StoppingRule:
    """What counts as improvement, and how long to wait for it.

    ``floor`` is the coverage inside the loop a head must reach before its
    checkpoints can be selected from. ``band`` picks which run-up width the
    objective is measured over. ``tolerance`` is how much better a step has to
    be to count as an improvement at all, and ``patience`` how many consecutive
    evaluations may pass without one.
    """

    floor: float = 0.3
    band: int = 256
    tolerance: float = 0.002
    patience: int = 4

    @property
    def objective(self) -> str:
        return f"warning_recall_{self.band}"


@dataclass(frozen=True)
class HeadOutcome:
    """What the rule decides for one depth."""

    layer: int
    became_eligible: bool
    stopped_at: Optional[int]
    selected_step: Optional[int]
    selected_value: Optional[float]
    best_in_pattern: float
    evaluations: int

    @property
    def selectable(self) -> bool:
        return self.selected_step is not None


def apply_rule(trajectory: pd.DataFrame, rule: StoppingRule, layer: int) -> HeadOutcome:
    """Walk one depth's evaluations in order and decide where it stops.

    Progress is tracked on whichever quantity the head still has to earn. Below
    the floor that is coverage inside the loop, since a head that has not found
    the loop yet sits near zero before it and would look flat on the objective
    while it was in fact still learning. At the floor the tracked quantity
    becomes the objective and the patience counter restarts, because what is
    being watched has changed.

    Eligibility latches. Coverage falls as well as rises, so a head near the
    floor would otherwise cross back and forth, restarting its counter each time
    and never stopping. Steps spent below the floor simply do not qualify for
    selection.
    """
    steps = trajectory.sort_values("step")
    objective, gate = rule.objective, "in_pattern_recall"
    if objective not in steps.columns or gate not in steps.columns:
        raise KeyError(f"trajectory is missing {objective!r} or {gate!r}")

    eligible = False
    waited = 0
    tracked_best = -np.inf
    selected_step: Optional[int] = None
    selected_value = -np.inf
    best_in_pattern = -np.inf
    stopped_at: Optional[int] = None
    seen = 0

    for row in steps.itertuples():
        seen += 1
        inside = float(getattr(row, gate))
        value = float(getattr(row, objective))
        best_in_pattern = max(best_in_pattern, inside)

        if not eligible and inside >= rule.floor:
            # The head has earned the objective. Watch that from here, and give
            # it a fresh patience budget to improve on it.
            eligible = True
            waited = 0
            tracked_best = -np.inf

        tracked = value if eligible else inside
        if eligible and value > selected_value:
            selected_step, selected_value = int(row.step), value

        if tracked > tracked_best + rule.tolerance:
            tracked_best = tracked
            waited = 0
        else:
            waited += 1
            if waited >= rule.patience:
                stopped_at = int(row.step)
                break

    return HeadOutcome(
        layer=layer,
        became_eligible=eligible,
        stopped_at=stopped_at,
        selected_step=selected_step,
        selected_value=None if selected_step is None else float(selected_value),
        best_in_pattern=float(best_in_pattern),
        evaluations=seen,
    )


def apply_rule_to_run(history: pd.DataFrame, rule: StoppingRule) -> pd.DataFrame:
    """Every depth of one run, decided independently.

    A run ends when its last depth stops, so the cost of a run is set by the
    slowest depth rather than the typical one. That maximum is reported here
    because it is what a future run's walltime has to cover.
    """
    rows = []
    for layer, trajectory in history.groupby("layer"):
        outcome = apply_rule(trajectory, rule, int(layer))
        rows.append(
            {
                "layer": outcome.layer,
                "became_eligible": outcome.became_eligible,
                "stopped_at": outcome.stopped_at,
                "selected_step": outcome.selected_step,
                "selected_value": outcome.selected_value,
                "best_in_pattern": outcome.best_in_pattern,
                "evaluations": outcome.evaluations,
            }
        )
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def run_length(outcomes: pd.DataFrame, cap: int) -> int:
    """The step a run would have reached: its slowest depth, or the cap."""
    stops = outcomes["stopped_at"].dropna()
    if len(stops) < len(outcomes):
        return int(cap)
    return int(min(cap, stops.max()))


def sweep(
    history: pd.DataFrame,
    *,
    floors: Sequence[float],
    bands: Sequence[int],
    tolerances: Sequence[float],
    patiences: Sequence[int],
    cap: int,
) -> pd.DataFrame:
    """What each setting of the rule would have decided, over one run.

    The four numbers are not guessable and are not worth guessing: every
    combination is a group-by over records that already exist, so they are read
    off rather than chosen.
    """
    rows = []
    for floor in floors:
        for band in bands:
            for tolerance in tolerances:
                for patience in patiences:
                    rule = StoppingRule(
                        floor=floor, band=band, tolerance=tolerance, patience=patience
                    )
                    outcomes = apply_rule_to_run(history, rule)
                    usable = outcomes[outcomes["selected_step"].notna()]
                    rows.append(
                        {
                            "floor": floor,
                            "band": band,
                            "tolerance": tolerance,
                            "patience": patience,
                            "depths eligible": int(outcomes["became_eligible"].sum()),
                            "depths selectable": len(usable),
                            "run stops at": run_length(outcomes, cap),
                            "best depth": (
                                int(usable.loc[usable["selected_value"].idxmax(), "layer"])
                                if len(usable)
                                else None
                            ),
                            "best value": (
                                float(usable["selected_value"].max()) if len(usable) else None
                            ),
                        }
                    )
    return pd.DataFrame(rows)
