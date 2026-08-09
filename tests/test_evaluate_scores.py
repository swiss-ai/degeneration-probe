"""The reporting entry point, in particular the boundary it enforces.

Choosing a threshold on the split being reported is the one mistake that
cannot be spotted in the output, because the numbers look better rather than
wrong. So it is made impossible rather than discouraged, and that is what these
check.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from degeneration_probe.evaluation.scores import build_scores, write_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
LENGTH, ONSET = 60, 30


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate_scores = _load_script("evaluate_scores")


def _scored_split(run_dir, split, *, positives=12, negatives=40):
    rng = np.random.default_rng(0)
    records = []
    for index in range(positives):
        scores = rng.uniform(0.0, 0.3, LENGTH).astype(np.float32)
        scores[ONSET:] = rng.uniform(0.7, 1.0, LENGTH - ONSET)
        records.append(
            {
                "prompt_id": f"{split}_p{index}",
                "rollout_idx": 0,
                "domain": "deepmath_103k" if index % 2 else "aime_2025",
                "split": split,
                "stop_reason": "length",
                "num_tokens": LENGTH,
                "onset_position": float(ONSET),
                "is_positive": True,
                "scores": scores,
            }
        )
    for index in range(negatives):
        records.append(
            {
                "prompt_id": f"{split}_n{index}",
                "rollout_idx": 0,
                "domain": "deepmath_103k" if index % 2 else "aime_2025",
                "split": split,
                "stop_reason": "eos",
                "num_tokens": LENGTH,
                "onset_position": None,
                "is_positive": False,
                "scores": rng.uniform(0.0, 0.4, LENGTH).astype(np.float32),
            }
        )
    write_scores(build_scores(records), run_dir / "scores" / f"{split}.parquet")


@pytest.fixture
def run_dir(tmp_path):
    _scored_split(tmp_path, "val")
    _scored_split(tmp_path, "test_indomain")
    return tmp_path


def test_thresholds_are_frozen_from_validation_and_read_back_unchanged(run_dir):
    frozen = evaluate_scores.freeze_thresholds(run_dir, [0.01, 0.05, 0.10], persistence=1)
    assert [t.target_negative_fpr for t in frozen] == [0.01, 0.05, 0.10]
    assert all(t.realized_negative_fpr <= t.target_negative_fpr + 1e-9 for t in frozen)

    reloaded, persistence = evaluate_scores.load_frozen_thresholds(run_dir)
    assert persistence == 1
    assert [t.tau for t in reloaded] == [t.tau for t in frozen]


def test_a_test_split_cannot_be_reported_without_frozen_thresholds(run_dir):
    with pytest.raises(FileNotFoundError, match="frozen before any other split"):
        evaluate_scores.load_frozen_thresholds(run_dir)


def test_reporting_writes_every_view_plus_a_summary(run_dir):
    thresholds = evaluate_scores.freeze_thresholds(run_dir, [0.05], persistence=1)
    out = evaluate_scores.report_split(run_dir, "test_indomain", thresholds, 1, 10)
    written = {path.name for path in out.iterdir()}
    assert {
        "view_a_detection.csv",
        "view_b_coverage.csv",
        "view_c_lead_time.csv",
        "view_d_persistence.csv",
        "per_domain_detection.csv",
        "rollout_table.csv",
        "summary.json",
    } <= written


def test_the_test_split_is_scored_at_the_threshold_validation_chose(run_dir):
    import pandas as pd

    thresholds = evaluate_scores.freeze_thresholds(run_dir, [0.05], persistence=1)
    out = evaluate_scores.report_split(run_dir, "test_indomain", thresholds, 1, 10)
    reported = pd.read_csv(out / "view_a_detection.csv")
    assert reported["tau"].iloc[0] == pytest.approx(thresholds[0].tau)


def test_the_persistence_trade_off_is_tabulated_on_validation_only(run_dir):
    comparison = evaluate_scores.compare_persistence(run_dir, [0.05], [1, 3, 5])
    assert sorted(comparison["persistence"].unique()) == [1, 3, 5]
    assert len(comparison) == 3
    for column in ("median_offset", "false_early_stop_rate", "positive_duty_cycle"):
        assert column in comparison.columns


def test_a_missing_score_table_says_which_step_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run score_rollouts.py first"):
        evaluate_scores.freeze_thresholds(tmp_path, [0.05], persistence=1)
