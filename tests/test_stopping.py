"""The rule applied to a run as it trains.

What matters here is not that a run stops, but that it stops on the same terms
a replay of its saved checkpoints would have used. The decision itself is
covered by ``test_head_selection``; this covers the bookkeeping around it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from transformers import TrainerControl, TrainerState

from degeneration_probe.evaluation.head_selection import StoppingRule, validation_record
from degeneration_probe.training.stopping import (
    RECORD_FIELDS,
    SELECTION_HISTORY_FILE,
    SELECTION_OUTCOMES_FILE,
    HeadSelection,
    records_from_metrics,
)

RULE = StoppingRule(floor=0.3, band=256, tolerance=0.002, patience=2)


def _refuse(literal):
    raise AssertionError(f"{literal} is not JSON any other reader will accept")


def test_the_recorded_columns_are_the_ones_the_rule_reads():
    """The trainer writes what the replay writes, so one analysis reads both."""
    record = validation_record(
        negative_peaks=[0.1, 0.2, 0.3],
        positive_scores=[np.concatenate([np.zeros(4), np.ones(4)])],
        onsets=[4],
    )
    assert set(RECORD_FIELDS) == set(record)


def test_a_single_depth_run_reports_without_a_namespace():
    plain = {"val/warning_recall_256": 0.02, "val/in_pattern_recall": 0.5}
    assert records_from_metrics(plain, prefix="val", layers=[12]) == [
        {"layer": 12, "in_pattern_recall": 0.5, "warning_recall_256": 0.02}
    ]
    # With several depths there is no unambiguous owner for an unprefixed value,
    # and the unprefixed one is a copy of the best depth's row in any case.
    assert records_from_metrics(plain, prefix="val", layers=[4, 12]) == []


def _drive(callback, control, values, *, start=50, every=50):
    """Feed one depth's trajectory in, one evaluation at a time."""
    state = TrainerState()
    for index, (inside, objective) in enumerate(values):
        state.global_step = start + index * every
        callback.on_evaluate(
            None,
            state,
            control,
            metrics={
                "val/in_pattern_recall": inside,
                "val/warning_recall_256": objective,
                "val/warning_recall_128": objective,
                "val/budget_tau": 0.9,
                "val/budget_realized_fpr": 0.01,
                "val/recall_at_budget": 0.5,
                "val/never_fired_positives": 0.0,
                "val/median_offset": -10.0,
            },
        )
    return state


def test_a_run_ends_once_its_last_depth_has_stopped(tmp_path):
    callback = HeadSelection(tmp_path, rule=RULE, layers=[12])
    control = TrainerControl()
    # Above the floor from the start, improving twice and then flat.
    _drive(callback, control, [(0.5, 0.01), (0.5, 0.02), (0.5, 0.02), (0.5, 0.02)])
    assert control.should_training_stop is True

    history = pd.read_parquet(tmp_path / SELECTION_HISTORY_FILE)
    assert list(history["step"]) == [50, 100, 150, 200]
    # Strictly, so that a depth with no selected step is spelled null rather
    # than as a bare NaN that only Python's own reader accepts.
    verdict = json.loads(
        (tmp_path / SELECTION_OUTCOMES_FILE).read_text(), parse_constant=_refuse
    )
    assert verdict["rule"]["objective"] == "warning_recall_256"
    assert verdict["depths"][0]["selected_step"] == 100


def test_a_still_improving_run_is_left_alone(tmp_path):
    callback = HeadSelection(tmp_path, rule=RULE, layers=[12])
    control = TrainerControl()
    _drive(callback, control, [(0.5, 0.01), (0.5, 0.02), (0.5, 0.03), (0.5, 0.04)])
    assert control.should_training_stop is False


def test_a_run_ends_only_when_every_depth_has(tmp_path):
    """A run's cost is set by its slowest depth, not by its typical one."""
    callback = HeadSelection(tmp_path, rule=RULE, layers=[4, 12])
    control = TrainerControl()
    state = TrainerState()
    flat = [0.02, 0.02, 0.02, 0.02, 0.02]
    climbing = [0.01, 0.02, 0.03, 0.04, 0.05]
    for index in range(len(flat)):
        state.global_step = 50 + index * 50
        callback.on_evaluate(
            None,
            state,
            control,
            metrics={
                "val/layer04/in_pattern_recall": 0.5,
                "val/layer04/warning_recall_256": flat[index],
                "val/layer12/in_pattern_recall": 0.5,
                "val/layer12/warning_recall_256": climbing[index],
            },
        )
    # The flat depth stopped long ago; the climbing one holds the run open.
    assert control.should_training_stop is False
    outcomes = callback.outcomes.set_index("layer")
    assert outcomes.loc[4, "stopped_at"] == 150
    assert pd.isna(outcomes.loc[12, "stopped_at"])


def test_the_rule_can_be_measured_without_being_obeyed(tmp_path):
    """Switching stopping off still leaves the trajectory behind."""
    callback = HeadSelection(tmp_path, rule=RULE, layers=[12], stop_when_finished=False)
    control = TrainerControl()
    _drive(callback, control, [(0.5, 0.02)] * 4)
    assert control.should_training_stop is False
    assert (tmp_path / SELECTION_HISTORY_FILE).is_file()


def test_a_re_evaluated_step_does_not_vote_twice(tmp_path):
    """Resuming re-scores the step it resumed from, which is not new evidence."""
    callback = HeadSelection(tmp_path, rule=RULE, layers=[12])
    control = TrainerControl()
    _drive(callback, control, [(0.5, 0.01), (0.5, 0.02)])
    state = TrainerState()
    state.global_step = 100
    callback.on_evaluate(
        None,
        state,
        control,
        metrics={"val/in_pattern_recall": 0.5, "val/warning_recall_256": 0.02},
    )
    history = pd.read_parquet(tmp_path / SELECTION_HISTORY_FILE)
    assert list(history["step"]) == [50, 100]


def test_a_depth_that_never_became_selectable_is_written_as_absent(tmp_path):
    """Its step is missing, not a number no reader outside Python accepts."""
    callback = HeadSelection(tmp_path, rule=RULE, layers=[12])
    # Never anywhere near the floor, and flat, so it stops without selecting.
    _drive(callback, callback_control := TrainerControl(), [(0.01, 0.0)] * 4)
    assert callback_control.should_training_stop is True
    verdict = json.loads(
        (tmp_path / SELECTION_OUTCOMES_FILE).read_text(), parse_constant=_refuse
    )
    depth = verdict["depths"][0]
    assert depth["became_eligible"] is False
    assert depth["selected_step"] is None
    assert depth["selected_value"] is None
