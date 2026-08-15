"""The per-depth stopping rule, checked against trajectories written by hand.

A stopping rule fails quietly: it picks a plausible step for the wrong reason
and nothing downstream looks wrong. So every case here is a trajectory whose
right answer can be worked out on paper, including the ones that only show up
rarely: a head that plateaus without ever learning, a head that crosses the
floor and falls back, and a head that improves forever.
"""

import numpy as np
import pandas as pd
import pytest

from degeneration_probe.evaluation.head_selection import (
    StoppingRule,
    apply_rule,
    apply_rule_to_run,
    run_length,
    validation_record,
)

RULE = StoppingRule(floor=0.3, band=256, tolerance=0.0, patience=2)


def trajectory(inside, warning, step=50):
    """One depth's evaluations, in order."""
    return pd.DataFrame(
        {
            "step": [(i + 1) * step for i in range(len(inside))],
            "in_pattern_recall": inside,
            "warning_recall_256": warning,
        }
    )


# --- the record ----------------------------------------------------------------


def test_the_threshold_comes_from_the_healthy_rollouts_only():
    # Ten healthy rollouts at a 10% budget allows one, so the threshold is the
    # second highest of them and nothing about the degenerate ones moves it.
    negatives = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    record = validation_record(
        negative_peaks=negatives,
        positive_scores=[np.ones(10)],
        onsets=[5],
        budget=0.10,
    )
    assert record["budget_tau"] == pytest.approx(0.9, abs=1e-6)
    assert record["recall_at_budget"] == 1.0


def test_coverage_splits_at_the_frontier_and_the_band_sits_before_it():
    # Fires from token 40 of a 100-token rollout whose loop starts at 60.
    scores = np.zeros(100)
    scores[40:] = 1.0
    record = validation_record(
        negative_peaks=[0.0] * 100,
        positive_scores=[scores],
        onsets=[60],
        budget=0.01,
        bands=(20,),
    )
    assert record["in_pattern_recall"] == 1.0
    assert record["warning_recall_20"] == 1.0          # tokens 40-59, all flagged
    assert record["median_offset"] == -20.0


def test_a_rollout_that_is_never_flagged_is_counted_and_lowers_coverage():
    hit = np.concatenate([np.zeros(60), np.ones(40)])
    miss = np.zeros(100)
    record = validation_record(
        negative_peaks=[0.5] * 100,
        positive_scores=[hit, miss],
        onsets=[60, 60],
        budget=0.01,
        bands=(20,),
    )
    assert record["never_fired_positives"] == 1.0
    # The missed rollout contributes zeros to the numerator and its full share
    # to the denominator, rather than dropping out the way it does from a median.
    assert record["in_pattern_recall"] == pytest.approx(0.5)


# --- stopping ------------------------------------------------------------------


def test_a_head_that_plateaus_below_the_floor_still_stops():
    """The case that would otherwise hold a whole run open.

    Coverage inside the loop never reaches the floor, so the head never becomes
    selectable. It must still stop once it has stopped improving, because a run
    ends only when its slowest depth does.
    """
    outcome = apply_rule(trajectory([0.05, 0.10, 0.10, 0.10, 0.10], [0.0] * 5), RULE, 12)
    assert outcome.became_eligible is False
    assert outcome.stopped_at == 200          # two flat evaluations after step 100
    assert outcome.selected_step is None      # and nothing is selected from it
    assert outcome.selectable is False


def test_below_the_floor_progress_is_judged_on_getting_to_the_loop():
    # Coverage inside the loop climbs while coverage before it sits at zero. A
    # rule watching the objective here would stop a head that is still learning.
    outcome = apply_rule(
        trajectory([0.05, 0.12, 0.20, 0.26, 0.31, 0.34], [0.0, 0.0, 0.0, 0.0, 0.01, 0.02]),
        RULE,
        12,
    )
    assert outcome.stopped_at is None
    assert outcome.became_eligible is True


def test_patience_restarts_when_the_head_crosses_the_floor():
    # Flat before the loop for the first two evaluations, which would exhaust a
    # patience of two, but the head is still below the floor and climbing, so
    # those do not count against the objective.
    outcome = apply_rule(
        trajectory([0.10, 0.20, 0.35, 0.40, 0.42], [0.00, 0.00, 0.05, 0.09, 0.13]),
        RULE,
        12,
    )
    assert outcome.stopped_at is None
    assert outcome.selected_step == 250
    assert outcome.selected_value == pytest.approx(0.13)


def test_selection_ignores_steps_taken_before_the_floor_was_reached():
    # The highest coverage before the loop happens at step 50, while the head is
    # still below the floor and its number means nothing.
    outcome = apply_rule(
        trajectory([0.05, 0.35, 0.40, 0.42], [0.90, 0.05, 0.06, 0.07]), RULE, 12
    )
    assert outcome.selected_step == 200
    assert outcome.selected_value == pytest.approx(0.07)


def test_eligibility_latches_so_a_head_near_the_floor_cannot_oscillate():
    # Crosses the floor, falls back under it, comes back. If eligibility were
    # re-evaluated each step the counter would restart on every crossing and the
    # head would never stop.
    outcome = apply_rule(
        trajectory([0.31, 0.28, 0.32, 0.29, 0.31], [0.10, 0.10, 0.10, 0.10, 0.10]),
        RULE,
        12,
    )
    assert outcome.became_eligible is True
    assert outcome.stopped_at == 150


def test_a_head_that_keeps_improving_never_stops():
    outcome = apply_rule(
        trajectory([0.4] * 6, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), RULE, 12
    )
    assert outcome.stopped_at is None
    assert outcome.selected_step == 300


def test_tolerance_makes_a_small_gain_count_as_no_gain():
    rising = trajectory([0.4] * 5, [0.10, 0.1005, 0.1010, 0.1015, 0.1020])
    assert apply_rule(rising, StoppingRule(0.3, 256, 0.0, 2), 12).stopped_at is None
    assert apply_rule(rising, StoppingRule(0.3, 256, 0.01, 2), 12).stopped_at == 150


def test_a_run_lasts_as_long_as_its_slowest_depth():
    history = pd.concat(
        [
            trajectory([0.4] * 5, [0.1, 0.2, 0.2, 0.2, 0.2]).assign(layer=1),
            trajectory([0.4] * 5, [0.1, 0.2, 0.3, 0.4, 0.5]).assign(layer=2),
        ]
    )
    outcomes = apply_rule_to_run(history, RULE)
    assert list(outcomes["layer"]) == [1, 2]
    # Depth 1 stops at 200; depth 2 never does, so the cap decides the run.
    assert outcomes.loc[outcomes.layer == 1, "stopped_at"].iloc[0] == 200
    assert pd.isna(outcomes.loc[outcomes.layer == 2, "stopped_at"].iloc[0])
    assert run_length(outcomes, cap=2000) == 2000


def test_a_run_ends_early_only_when_every_depth_has_stopped():
    history = pd.concat(
        [
            trajectory([0.4] * 5, [0.1, 0.2, 0.2, 0.2, 0.2]).assign(layer=1),
            trajectory([0.4] * 5, [0.1, 0.2, 0.3, 0.3, 0.3]).assign(layer=2),
        ]
    )
    outcomes = apply_rule_to_run(history, RULE)
    assert run_length(outcomes, cap=2000) == 250


# --- the identity the replay rests on ------------------------------------------


def test_a_trained_norm_and_a_linear_map_collapse_into_one_linear_map():
    """Folding the normalization into the head must change nothing.

    The replay scores every checkpoint of a depth with one matrix multiply,
    which is only allowed because a trained scale and shift followed by a linear
    map is another linear map. If that stops being true the replayed numbers
    drift from what training would have recorded, silently.
    """
    import torch

    from degeneration_probe.probes.linear_probe import apply_head, build_normalization

    torch.manual_seed(0)
    hidden, tokens = 64, 17
    states = torch.randn(1, tokens, hidden)

    norm = build_normalization("layernorm", hidden, device=torch.device("cpu"), dtype=torch.float32)
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.2)
    torch.nn.init.normal_(norm.bias, mean=0.0, std=0.2)
    linear = torch.nn.Linear(hidden, 1)

    direct = apply_head(states, pre_head_norm=norm, linear=linear, context_window_size=1)

    w = linear.weight.reshape(-1).detach()
    folded_w = w * norm.weight.detach()
    folded_b = float(linear.bias.detach().reshape(-1)[0]) + float(
        torch.dot(w, norm.bias.detach())
    )
    normalised = torch.nn.functional.layer_norm(states, (hidden,), eps=1e-5)
    collapsed = normalised @ folded_w + folded_b

    assert torch.allclose(direct, collapsed, atol=1e-5)
