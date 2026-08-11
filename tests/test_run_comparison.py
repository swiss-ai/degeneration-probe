"""Pooling seeds, and judging a rung difference against its noise floor."""

import json

import numpy as np
import pandas as pd
import pytest

from degeneration_probe.analysis.run_comparison import (
    collect_results,
    collect_runs,
    ladder_deltas,
    pool_seeds,
)


def _write_run(root, group, seed, precision, recall=0.9, offset=-50.0):
    run_dir = root / f"{group}_s{seed}" / "20260101T000000"
    (run_dir / "evaluation" / "test_indomain").mkdir(parents=True)
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "run_name": f"{group}_s{seed}",
                "group": group,
                "status": "finished",
                "axes": {"seed": seed, "selection": group},
            }
        )
    )
    base = run_dir / "evaluation" / "test_indomain"
    pd.DataFrame(
        {"target_negative_fpr": [0.01, 0.05], "tau": [0.9, 0.5],
         "precision": [precision, precision - 0.1], "recall": [recall, recall],
         "f1": [0.5, 0.5], "negative_fpr": [0.01, 0.05], "positives": [100, 100]}
    ).to_csv(base / "view_a_detection.csv", index=False)
    pd.DataFrame(
        {"target_negative_fpr": [0.01, 0.05], "median_offset": [offset, offset - 20],
         "never_fired_positives": [0, 0], "false_early_stop_rate": [0.01, 0.05]}
    ).to_csv(base / "view_c_lead_time.csv", index=False)
    (base / "summary.json").write_text(json.dumps({"rank_metrics": {"rollout_auc": 0.99}}))
    return run_dir


@pytest.fixture
def outputs(tmp_path):
    for seed, precision in zip((1, 2, 3), (0.80, 0.82, 0.84)):
        _write_run(tmp_path, "rung_a", seed, precision)
    for seed, precision in zip((1, 2, 3), (0.90, 0.92, 0.94)):
        _write_run(tmp_path, "rung_b", seed, precision)
    return tmp_path


def test_runs_are_grouped_by_recipe_not_by_seed(outputs):
    runs = collect_runs(outputs)
    assert len(runs) == 6
    assert set(runs["group"]) == {"rung_a", "rung_b"}
    assert sorted(runs[runs["group"] == "rung_a"]["seed"]) == [1, 2, 3]


def test_pooling_reports_a_spread_alongside_every_mean(outputs):
    pooled = pool_seeds(collect_results(collect_runs(outputs), "test_indomain"))
    row = pooled[(pooled["group"] == "rung_a") & (pooled["target_negative_fpr"] == 0.01)].iloc[0]
    assert row["seeds"] == 3
    assert row["precision_mean"] == pytest.approx(0.82)
    assert row["precision_std"] == pytest.approx(np.std([0.80, 0.82, 0.84], ddof=1))
    # The lead time is pooled from the lead-time view, not the detection one.
    assert row["median_offset_mean"] == pytest.approx(-50.0)


def test_a_difference_is_reported_against_the_spread_it_came_from(outputs):
    pooled = pool_seeds(collect_results(collect_runs(outputs), "test_indomain"))
    deltas = ladder_deltas(pooled, ["rung_a", "rung_b"])
    row = deltas[deltas["target_negative_fpr"] == 0.01].iloc[0]
    assert row["from"] == "rung_a" and row["to"] == "rung_b"
    assert row["precision_delta"] == pytest.approx(0.10)
    # Clearly larger than the seed-to-seed variation, so it is a result.
    assert row["precision_beats_noise"]


def test_a_difference_inside_the_noise_is_not_called_a_result(tmp_path):
    for seed, precision in zip((1, 2, 3), (0.70, 0.85, 0.95)):
        _write_run(tmp_path, "noisy_a", seed, precision)
    for seed, precision in zip((1, 2, 3), (0.72, 0.87, 0.97)):
        _write_run(tmp_path, "noisy_b", seed, precision)
    pooled = pool_seeds(collect_results(collect_runs(tmp_path), "test_indomain"))
    deltas = ladder_deltas(pooled, ["noisy_a", "noisy_b"])
    row = deltas[deltas["target_negative_fpr"] == 0.01].iloc[0]
    assert row["precision_delta"] == pytest.approx(0.02, abs=1e-6)
    assert not row["precision_beats_noise"]


def test_only_adjacent_rungs_are_compared(outputs):
    for seed, precision in zip((1, 2, 3), (0.95, 0.96, 0.97)):
        _write_run(outputs, "rung_c", seed, precision)
    pooled = pool_seeds(collect_results(collect_runs(outputs), "test_indomain"))
    deltas = ladder_deltas(pooled, ["rung_a", "rung_b", "rung_c"])
    pairs = set(zip(deltas["from"], deltas["to"]))
    assert pairs == {("rung_a", "rung_b"), ("rung_b", "rung_c")}
