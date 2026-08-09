"""The protocol checked against scorers whose answers are known by hand.

An off-by-one in the alarm position, or in the split at the frontier, produces
numbers that look entirely plausible. The only defence is a set of scorers
whose every view can be worked out on paper: a perfect oracle, one that fires
late, one that fires early, a constant, pure noise, and one that spikes for a
single token on healthy text.
"""

import numpy as np
import pandas as pd
import pytest

from degeneration_probe.evaluation.protocol import (
    Threshold,
    alarm_staircase,
    choose_thresholds,
    evaluate,
    first_alarm,
    per_rollout_table,
    persistence_scores,
    rollout_score,
    threshold_for_budget,
    view_a_detection,
    view_b_coverage,
    view_c_lead_time,
    view_d_persistence,
)
from degeneration_probe.evaluation.scores import build_scores

LENGTH = 100
ONSET = 60
HALF = Threshold(target_negative_fpr=0.0, tau=0.5, realized_negative_fpr=0.0, negative_rollouts=0)


def _rollout(name, scores, *, onset=None, domain="d", split="val"):
    scores = np.asarray(scores, dtype=np.float32)
    return {
        "prompt_id": name,
        "rollout_idx": 0,
        "domain": domain,
        "split": split,
        "stop_reason": "length" if onset is not None else "eos",
        "num_tokens": int(scores.size),
        "onset_position": float(onset) if onset is not None else None,
        "is_positive": onset is not None,
        "scores": scores,
    }


def oracle(shift=0):
    """Fires exactly at the frontier, shifted by a known number of tokens."""
    scores = np.zeros(LENGTH, dtype=np.float32)
    scores[max(ONSET + shift, 0) :] = 1.0
    return scores


def _population(positive_scores, negative_scores, split="val"):
    records = [
        _rollout(f"pos{i}", s, onset=ONSET, split=split)
        for i, s in enumerate(positive_scores)
    ]
    records += [_rollout(f"neg{i}", s, split=split) for i, s in enumerate(negative_scores)]
    return build_scores(records)


# --- the alarm itself ----------------------------------------------------------


def test_the_alarm_is_the_first_token_at_or_above_the_threshold():
    assert first_alarm(oracle(), 0.5) == ONSET
    assert first_alarm(oracle(shift=+10), 0.5) == ONSET + 10
    assert first_alarm(oracle(shift=-25), 0.5) == ONSET - 25
    assert first_alarm(np.zeros(LENGTH), 0.5) is None
    # At the threshold exactly, not just above it.
    assert first_alarm(np.array([0.0, 0.5, 1.0]), 0.5) == 1


def test_persistence_requires_consecutive_tokens_and_costs_lead_time():
    spiky = np.zeros(LENGTH, dtype=np.float32)
    spiky[10] = 1.0  # a lone spike
    spiky[50:] = 1.0  # a sustained run
    assert first_alarm(spiky, 0.5, persistence=1) == 10
    assert first_alarm(spiky, 0.5, persistence=3) == 50
    # A sustained alarm is delayed by nothing; the window closes on its start.
    assert first_alarm(oracle(), 0.5, persistence=5) == ONSET
    # A run shorter than the window can never trip the rule.
    brief = np.zeros(20, dtype=np.float32)
    brief[5:8] = 1.0
    assert first_alarm(brief, 0.5, persistence=3) == 5
    assert first_alarm(brief, 0.5, persistence=4) is None


def test_a_rollout_shorter_than_the_window_can_never_fire():
    assert persistence_scores(np.ones(3), 5).size == 0
    assert first_alarm(np.ones(3), 0.5, persistence=5) is None
    assert rollout_score(np.ones(3), persistence=5) == 0.0


def test_one_pass_answers_every_threshold():
    scores = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.95], dtype=np.float32)
    positions, levels = alarm_staircase(scores)
    assert list(positions) == [0, 1, 3, 5]
    assert np.allclose(levels, [0.1, 0.4, 0.9, 0.95], atol=1e-6)
    # The staircase agrees with a direct scan at every threshold.
    for tau in np.linspace(0, 1, 21):
        direct = next((i for i, s in enumerate(scores) if s >= tau), None)
        assert first_alarm(scores, tau) == direct


def test_the_ranking_score_is_exactly_what_fires_at_a_threshold():
    scores = np.array([0.1, 0.7, 0.3], dtype=np.float32)
    peak = rollout_score(scores)
    assert peak == pytest.approx(0.7)
    assert (first_alarm(scores, 0.7) is not None) is (peak >= 0.7)
    assert (first_alarm(scores, 0.71) is not None) is (peak >= 0.71)


# --- thresholds ----------------------------------------------------------------


def test_a_budget_buys_exactly_the_false_alarms_it_pays_for():
    scores = np.linspace(0.0, 1.0, 100)
    for budget in (0.0, 0.01, 0.05, 0.10, 0.5):
        tau, realized = threshold_for_budget(scores, budget)
        assert realized <= budget + 1e-9
        assert (scores >= tau).mean() == pytest.approx(realized)
    tau, realized = threshold_for_budget(scores, 1.0)
    assert realized == 1.0


def test_thresholds_may_only_be_chosen_on_validation_data():
    frame = _population([oracle()], [np.zeros(LENGTH)], split="test_indomain")
    with pytest.raises(ValueError, match="only be chosen on 'val'"):
        choose_thresholds(frame, [0.05])


def test_choosing_a_threshold_needs_negatives_to_measure_against():
    frame = _population([oracle()], [], split="val")
    with pytest.raises(ValueError, match="negative rollouts"):
        choose_thresholds(frame, [0.05])


# --- the views on a perfect scorer ---------------------------------------------


def test_a_perfect_oracle_scores_perfectly_in_every_view():
    frame = _population([oracle()] * 5, [np.zeros(LENGTH)] * 20)
    result = evaluate(frame, [HALF])

    detection = result["view_a_detection"].iloc[0]
    assert (detection.true_positive, detection.false_positive) == (5, 0)
    assert (detection.precision, detection.recall) == (1.0, 1.0)
    assert result["rank_metrics"]["rollout_auc"] == 1.0

    coverage = result["view_b_coverage"].iloc[0]
    assert coverage.token_false_positive_rate == 0.0
    assert coverage.in_pattern_recall == 1.0
    # Every token of every negative rollout is counted, with no subsampling.
    assert coverage.negative_tokens == 20 * LENGTH
    assert coverage.in_pattern_tokens == 5 * (LENGTH - ONSET)
    assert coverage.pre_frontier_tokens == 5 * ONSET

    lead = result["view_c_lead_time"].iloc[0]
    assert lead.median_offset == 0.0
    assert lead.never_fired_positives == 0
    assert lead.false_early_stop_rate == 0.0


def test_lead_time_is_signed_so_early_and_late_are_distinguishable():
    early = _population([oracle(shift=-20)] * 3, [np.zeros(LENGTH)] * 5)
    late = _population([oracle(shift=+15)] * 3, [np.zeros(LENGTH)] * 5)
    assert view_c_lead_time(per_rollout_table(early, [HALF])).iloc[0].median_offset == -20.0
    assert view_c_lead_time(per_rollout_table(late, [HALF])).iloc[0].median_offset == +15.0
    assert view_c_lead_time(per_rollout_table(early, [HALF])).iloc[0].fired_before_frontier == 1.0


def test_a_scorer_that_never_fires_is_counted_apart_from_the_offsets():
    frame = _population([oracle(), np.zeros(LENGTH)], [np.zeros(LENGTH)] * 4)
    lead = view_c_lead_time(per_rollout_table(frame, [HALF])).iloc[0]
    assert lead.detected_positives == 1
    assert lead.never_fired_positives == 1
    # The miss does not drag the offset of the rollout that was caught.
    assert lead.median_offset == 0.0


def test_a_constant_scorer_looks_perfect_until_the_false_alarms_are_read():
    frame = _population([np.ones(LENGTH)] * 5, [np.ones(LENGTH)] * 20)
    result = evaluate(frame, [HALF])
    coverage = result["view_b_coverage"].iloc[0]
    persistence = result["view_d_persistence"]
    negative = persistence[persistence["population"] == "negative"].iloc[0]

    # Perfect recall and perfect persistence, and completely useless.
    assert coverage.in_pattern_recall == 1.0
    assert negative.mean_duty_cycle == 1.0
    assert coverage.token_false_positive_rate == 1.0
    assert result["view_c_lead_time"].iloc[0].false_early_stop_rate == 1.0
    assert np.isnan(result["rank_metrics"]["rollout_auc"]) or result["rank_metrics"][
        "rollout_auc"
    ] == pytest.approx(0.5)


# --- persistence ---------------------------------------------------------------


def test_persistence_separates_a_jittery_scorer_from_a_confident_one():
    spiky = np.zeros((10, LENGTH), dtype=np.float32)
    for index, row in enumerate(spiky):
        # Isolated by construction: adjacent spikes would legitimately survive
        # a two-token persistence window and blunt the comparison below.
        row[[10 + index, 40 + index, 70 + index]] = 1.0
    sustained = np.zeros((10, LENGTH), dtype=np.float32)
    sustained[:, 40:] = 1.0  # fires and holds

    jittery = view_d_persistence(per_rollout_table(_population([], list(spiky)), [HALF]))
    confident = view_d_persistence(per_rollout_table(_population([], list(sustained)), [HALF]))

    assert jittery.iloc[0].median_first_run_length == 1.0
    assert confident.iloc[0].median_first_run_length == LENGTH - 40
    assert jittery.iloc[0].median_episodes == 3.0
    assert confident.iloc[0].median_episodes == 1.0
    assert jittery.iloc[0].mean_duty_cycle < 0.2
    assert confident.iloc[0].mean_duty_cycle == 1.0
    assert jittery.iloc[0].stickiness < 0.1 < 0.9 < confident.iloc[0].stickiness
    # A scorer that keeps firing has not retracted, however jittery it is:
    # retraction is reserved for firing once and then falling silent for good.
    assert jittery.iloc[0].retraction_rate == 0.0
    assert confident.iloc[0].retraction_rate == 0.0

    # Which is what a persistence window is chosen from: m=2 removes every
    # single-token false alarm, and the sustained ones survive untouched.
    assert all(first_alarm(row, 0.5, persistence=2) is None for row in spiky)
    assert all(first_alarm(row, 0.5, persistence=2) == 40 for row in sustained)


def test_commitment_is_the_point_after_which_it_never_backs_down():
    dithering = np.zeros(LENGTH, dtype=np.float32)
    dithering[30] = 1.0
    dithering[35] = 1.0
    dithering[50:] = 1.0
    table = per_rollout_table(_population([], [dithering]), [HALF])
    row = table.iloc[0]
    assert row.alarm == 30
    assert row.first_run_length == 1.0
    assert row.episodes == 3
    assert row.tokens_to_commitment == 20.0  # fires at 30, settles at 50
    assert row.retracted == 0.0


def test_a_scorer_that_ends_below_the_threshold_never_commits():
    late_drop = np.ones(LENGTH, dtype=np.float32)
    late_drop[-1] = 0.0
    table = per_rollout_table(_population([], [late_drop]), [HALF])
    assert np.isinf(table.iloc[0].tokens_to_commitment)
    summary = view_d_persistence(table).iloc[0]
    assert summary.never_commits == 1
    assert np.isnan(summary.median_tokens_to_commitment)


# --- reporting rules -----------------------------------------------------------


def test_in_pattern_misses_are_counted_per_rollout_not_only_as_a_rate():
    holed = oracle()
    holed[[70, 71, 90]] = 0.0
    table = per_rollout_table(_population([holed], [np.zeros(LENGTH)]), [HALF])
    positive = table[table["is_positive"]].iloc[0]
    assert positive.in_pattern_tokens == LENGTH - ONSET
    assert positive.in_pattern_misses == 3


def test_underpowered_domains_are_marked_rather_than_quoted():
    records = [_rollout(f"a{i}", oracle(), onset=ONSET, domain="rich") for i in range(12)]
    records += [_rollout("b0", oracle(), onset=ONSET, domain="thin")]
    records += [_rollout(f"na{i}", np.zeros(LENGTH), domain="rich") for i in range(5)]
    records += [_rollout(f"nb{i}", np.zeros(LENGTH), domain="thin") for i in range(5)]
    result = evaluate(build_scores(records), [HALF])
    per_domain = result["per_domain_detection"].set_index("domain")
    assert bool(per_domain.loc["thin", "anecdotal"]) is True
    assert bool(per_domain.loc["rich", "anecdotal"]) is False


def test_the_frontier_splits_a_positive_rollout_at_the_right_token():
    # Fires on every pre-frontier token and nothing after: recall must be zero,
    # and the pre-frontier tokens must not be scored as positives.
    inverted = np.zeros(LENGTH, dtype=np.float32)
    inverted[:ONSET] = 1.0
    coverage = view_b_coverage(_population([inverted], []), [HALF]).iloc[0]
    assert coverage.in_pattern_recall == 0.0
    assert coverage.in_pattern_tokens == LENGTH - ONSET
    assert coverage.pre_frontier_tokens == ONSET


def test_retraction_is_firing_once_and_then_falling_silent_for_good():
    single_spike = np.zeros(LENGTH, dtype=np.float32)
    single_spike[20] = 1.0
    summary = view_d_persistence(per_rollout_table(_population([], [single_spike]), [HALF]))
    assert summary.iloc[0].retraction_rate == 1.0
    assert summary.iloc[0].median_episodes == 1.0


def test_a_zero_budget_holds_even_when_the_scorer_reaches_one():
    frame = _population([oracle()] * 3, [np.ones(LENGTH)] * 4)
    thresholds = choose_thresholds(frame, [0.0, 0.5])
    strict, loose = thresholds
    assert strict.realized_negative_fpr == 0.0
    assert strict.tau > 1.0
    assert loose.realized_negative_fpr <= 0.5
    # And nothing fires at the strict threshold, positives included.
    detection = view_a_detection(per_rollout_table(frame, [strict])).iloc[0]
    assert (detection.false_positive, detection.true_positive) == (0, 0)
