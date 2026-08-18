"""Filling in a warning width without repeating the replay.

The saving rests on two claims: the threshold each checkpoint was measured at is
fixed by the healthy answers, whose peaks were kept, and coverage of the
approach is a statement about the degenerate answers alone. Both are checked
here, along with the guard that stops a run being half-described by one scorer
and half by another.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backfill_warning_bands import check_against_record, recovered_thresholds
from degeneration_probe.evaluation.head_selection import STEERING_BUDGET, validation_record


def test_the_threshold_is_recovered_from_the_healthy_answers_alone():
    """No degenerate answer enters it, which is why they are the only re-read."""
    rng = np.random.default_rng(0)
    peaks = rng.random((500, 1, 1))
    scores = [np.concatenate([np.zeros(20), np.ones(30)])]
    record = validation_record(
        negative_peaks=peaks[:, 0, 0], positive_scores=scores, onsets=[20]
    )
    recovered = recovered_thresholds(peaks, STEERING_BUDGET)
    assert recovered[0, 0] == pytest.approx(record["budget_tau"], abs=1e-12)


def test_a_run_whose_threshold_has_moved_is_refused():
    """Half a table from one scorer and half from another is worse than none."""
    peaks = np.linspace(0, 1, 300).reshape(300, 1, 1)
    taus = recovered_thresholds(peaks, STEERING_BUDGET)
    recorded = pd.DataFrame(
        [{"step": 50, "layer": 12, "budget_tau": float(taus[0, 0])}]
    )
    check_against_record(recorded, taus, [50], [12])

    moved = recorded.assign(budget_tau=recorded["budget_tau"] + 1e-6)
    with pytest.raises(SystemExit, match="different scorers"):
        check_against_record(moved, taus, [50], [12])


def test_every_reported_width_is_measured():
    """A width nobody measured would read as an absence of warning."""
    from degeneration_probe.evaluation.protocol import WARNING_BANDS

    record = validation_record(
        negative_peaks=[0.1, 0.2, 0.3],
        positive_scores=[np.concatenate([np.zeros(300), np.ones(50)])],
        onsets=[300],
    )
    for band in WARNING_BANDS:
        assert f"warning_recall_{band}" in record
