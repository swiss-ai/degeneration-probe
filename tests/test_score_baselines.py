"""The heuristic baselines, mapped into the protocol's score contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "score_baselines", REPO_ROOT / "scripts" / "score_baselines.py"
)
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


def test_entropy_is_inverted_because_a_loop_is_confident():
    scores = baselines.entropy_scores([0.0, 1.0, 4.0], 3)
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] == pytest.approx(1.0)
    assert np.all((0 < scores) & (scores <= 1))


def test_the_entropy_transform_needs_no_population_to_normalize_against():
    # The same entropy maps to the same score whatever else was scored, so a
    # value does not change meaning with the split it came from.
    assert baselines.entropy_scores([2.0], 1)[0] == baselines.entropy_scores([2.0, 0.0], 2)[0]


def test_a_repeated_substring_becomes_a_step_where_the_repeat_begins():
    scores = baselines.lrs_scores(first_start=4, length=10, num_tokens=8)
    assert scores.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_a_rollout_with_no_repeat_never_fires():
    assert not baselines.lrs_scores(first_start=None, length=0, num_tokens=5).any()
    assert not baselines.lrs_scores(first_start=2, length=0, num_tokens=5).any()


def test_gaps_in_a_windowed_signal_are_carried_rather_than_left_undefined():
    # A windowed score is undefined for the first few tokens; the contract
    # requires a value at every position.
    scores = baselines.repetition_scores([np.nan, np.nan, 0.4, np.nan, 0.9], 5)
    assert np.isfinite(scores).all()
    assert scores.tolist() == [0.4, 0.4, 0.4, 0.4, 0.9]


def test_scores_stay_inside_the_unit_interval():
    for scores in (
        baselines.repetition_scores([-0.2, 0.5, 1.7], 3),
        baselines.entropy_scores([0.0, 12.0], 2),
        baselines.lrs_scores(1, 5, 4),
    ):
        assert np.all((scores >= 0) & (scores <= 1))
