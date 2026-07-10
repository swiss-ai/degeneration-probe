"""Tests for degeneration_probe.dataset_gen.label -- the per-token
repetition/LRS window scores, entropy pass-through, and prompt-level
aggregation.

All expected values below are worked by hand in the accompanying comments
(no dependency on the module under test to derive the expectation), following
the same "deterministic, no HF/network/GPU" style as test_generate.py /
test_repetition_converter.py. Loads config.py, paths.py, and label.py
directly by file path (bypassing degeneration_probe/__init__.py), the same
pattern used in test_generate.py, since label.py only needs numpy/pandas (no
torch/transformers).
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GEN_DIR = REPO_ROOT / "degeneration_probe" / "dataset_gen"


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dataset_gen():
    degeneration_probe_pkg = types.ModuleType("degeneration_probe")
    degeneration_probe_pkg.__path__ = [str(REPO_ROOT / "degeneration_probe")]
    sys.modules.setdefault("degeneration_probe", degeneration_probe_pkg)

    dataset_gen_pkg = types.ModuleType("degeneration_probe.dataset_gen")
    dataset_gen_pkg.__path__ = [str(DATASET_GEN_DIR)]
    sys.modules.setdefault("degeneration_probe.dataset_gen", dataset_gen_pkg)

    _load_module("degeneration_probe.dataset_gen.config", DATASET_GEN_DIR / "config.py")
    _load_module("degeneration_probe.dataset_gen.paths", DATASET_GEN_DIR / "paths.py")
    label_module = _load_module(
        "degeneration_probe.dataset_gen.label", DATASET_GEN_DIR / "label.py"
    )
    return label_module


label_module = _load_dataset_gen()

_ngram_ttr = label_module._ngram_ttr
sliding_window_repetition = label_module.sliding_window_repetition
lrs_score = label_module.lrs_score
sliding_window_lrs = label_module.sliding_window_lrs
label_rollout = label_module.label_rollout
label_shard = label_module.label_shard
aggregate_prompt_stats = label_module.aggregate_prompt_stats
_rollout_summary = label_module._rollout_summary


def _assert_list_allclose_with_nan(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        if isinstance(e, float) and math.isnan(e):
            assert isinstance(a, float) and math.isnan(a)
        else:
            assert a == pytest.approx(e)


# --- _ngram_ttr ----------------------------------------------------------------

def test_ngram_ttr_all_distinct_tokens_is_one():
    assert _ngram_ttr([1, 2, 3, 4], 1) == 1.0


def test_ngram_ttr_all_same_token_is_low():
    # 4 unigrams, only 1 distinct -> TTR = 1/4.
    assert _ngram_ttr([5, 5, 5, 5], 1) == pytest.approx(0.25)


def test_ngram_ttr_too_few_tokens_for_n_returns_one():
    assert _ngram_ttr([1, 2], 5) == 1.0


# --- sliding_window_repetition (repetition_score) -------------------------------

def test_sliding_window_repetition_hand_computed():
    # tokens[t:t+4] for t=0..4 (window=4, total=8):
    #   t=0: [1,1,1,1] -> 1 distinct/4 -> TTR=0.25 -> score=0.75
    #   t=1: [1,1,1,2] -> 2 distinct/4 -> TTR=0.5  -> score=0.5
    #   t=2: [1,1,2,3] -> 3 distinct/4 -> TTR=0.75 -> score=0.25
    #   t=3: [1,2,3,4] -> 4 distinct/4 -> TTR=1.0  -> score=0.0
    #   t=4: [2,3,4,5] -> 4 distinct/4 -> TTR=1.0  -> score=0.0
    #   t=5,6,7: window doesn't fit -> NaN
    tokens = [1, 1, 1, 1, 2, 3, 4, 5]
    got = sliding_window_repetition(tokens, window_size=4, n=1)
    expected = [0.75, 0.5, 0.25, 0.0, 0.0, float("nan"), float("nan"), float("nan")]
    _assert_list_allclose_with_nan(got, expected)


def test_sliding_window_repetition_no_repeats_is_zero_everywhere():
    # All-distinct tokens: every valid window has TTR=1.0 -> score=0.0.
    tokens = list(range(10))
    got = sliding_window_repetition(tokens, window_size=4, n=1)
    valid = got[:7]  # t=0..6 fit (end=t+4<=10)
    assert all(v == pytest.approx(0.0) for v in valid)
    assert all(math.isnan(v) for v in got[7:])


# --- lrs_score (whole-sequence, per the given reference) -----------------------

def test_lrs_score_obvious_repeated_block():
    # tokens = [1,2,3,1,2,3,7,8], min_length=2.
    # The block "1,2,3" at index 0 matches the one at index 3 (length 3,
    # non-overlapping since i+length=3 is not < j=3, so it stops there).
    # No longer match exists. max_len=3, n=8 -> score = 3/8.
    tokens = [1, 2, 3, 1, 2, 3, 7, 8]
    assert lrs_score(tokens, min_length=2) == pytest.approx(3 / 8)


def test_lrs_score_no_repeats_is_zero():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    assert lrs_score(tokens, min_length=2) == 0.0


def test_lrs_score_min_length_gates_close_repeats():
    # "1" repeats at index 0 and 1, but j=1 < i+min_length=3, so this repeat
    # is never considered (too close together) and there's no other match.
    tokens = [1, 1, 2, 3, 4, 5]
    assert lrs_score(tokens, min_length=3) == 0.0


# --- sliding_window_lrs (lrs_window_score): repeated block vs no repeat -------

def test_sliding_window_lrs_high_for_uniformly_repetitive_sequence():
    # 8 identical tokens, window=4, min_length=1: every window is 4 identical
    # tokens. By hand: for i=0, j=2 gives the longest non-overlapping match
    # (length 2, since i+length=2 is not < j=2 -- it stops there), and no
    # other (i, j) pair beats that. So lrs_score of any 4-identical-token
    # window is 2/4 = 0.5.
    tokens = [9] * 8
    got = sliding_window_lrs(tokens, window_size=4, min_length=1)
    valid = got[:5]  # t=0..4 fit (end=t+4<=8)
    assert all(v == pytest.approx(0.5) for v in valid)
    assert all(math.isnan(v) for v in got[5:])


def test_sliding_window_lrs_zero_for_non_repetitive_sequence():
    # All-distinct tokens: no window can contain a repeat -> lrs_score=0.
    tokens = list(range(8))
    got = sliding_window_lrs(tokens, window_size=4, min_length=1)
    valid = got[:5]
    assert all(v == pytest.approx(0.0) for v in valid)
    assert all(math.isnan(v) for v in got[5:])


def test_sliding_window_lrs_matches_direct_lrs_score_per_window():
    # Cross-check the windowing/NaN-masking wrapper against direct lrs_score
    # calls on the same windows (independent of the windowing logic itself).
    tokens = [1, 2, 3, 1, 2, 3, 7, 8, 9, 9]
    window_size = 5
    min_length = 2
    got = sliding_window_lrs(tokens, window_size=window_size, min_length=min_length)
    total = len(tokens)
    expected = []
    for t in range(total):
        end = t + window_size
        if end > total:
            expected.append(float("nan"))
        else:
            expected.append(lrs_score(tokens[t:end], min_length=min_length))
    _assert_list_allclose_with_nan(got, expected)


# --- label_rollout / label_shard ------------------------------------------------

def test_label_rollout_shapes_and_entropy_passthrough():
    token_ids = [1, 1, 1, 1, 2, 3, 4, 5]
    entropy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    labels = label_rollout(token_ids, entropy, window_size=4, ttr_ngram=1, lrs_min_length=2)

    assert len(labels["repetition_score"]) == len(token_ids)
    assert len(labels["lrs_window_score"]) == len(token_ids)
    assert len(labels["entropy"]) == len(token_ids)
    assert labels["entropy"] == pytest.approx(entropy)
    # Same hand-computed ttr as test_sliding_window_repetition_hand_computed.
    assert labels["repetition_score"][0] == pytest.approx(0.75)
    assert labels["repetition_score"][3] == pytest.approx(0.0)
    assert math.isnan(labels["repetition_score"][-1])


def test_label_shard_produces_one_row_per_rollout_with_expected_columns():
    generations_df = pd.DataFrame.from_records(
        [
            {
                "prompt_id": "p0",
                "rollout_idx": 0,
                "generated_token_ids": np.array([1, 1, 1, 1, 2, 3, 4, 5]),
                "per_token_entropy": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                "generated_text": "irrelevant",
            },
            {
                "prompt_id": "p0",
                "rollout_idx": 1,
                "generated_token_ids": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                "per_token_entropy": np.array([1.0] * 8),
                "generated_text": "irrelevant",
            },
        ]
    )
    labels_df = label_shard(generations_df, window_size=4, ttr_ngram=1, lrs_min_length=2)

    assert list(labels_df.columns) == ["prompt_id", "rollout_idx", "repetition_score", "lrs_window_score", "entropy"]
    assert len(labels_df) == 2
    assert set(labels_df["rollout_idx"]) == {0, 1}

    row0 = labels_df[labels_df["rollout_idx"] == 0].iloc[0]
    assert row0["repetition_score"][0] == pytest.approx(0.75)
    assert row0["entropy"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    row1 = labels_df[labels_df["rollout_idx"] == 1].iloc[0]
    # All-distinct tokens -> repetition score 0.0 everywhere a window fits.
    assert row1["repetition_score"][0] == pytest.approx(0.0)


# --- _rollout_summary / aggregate_prompt_stats ---------------------------------

def test_rollout_summary_ignores_nan_tail():
    repetition = [0.9, float("nan")]
    lrs = [0.1, float("nan")]
    summary = _rollout_summary(repetition, lrs)
    assert summary["mean_repetition_score"] == pytest.approx(0.9)
    assert summary["max_repetition_score"] == pytest.approx(0.9)
    assert summary["mean_lrs_window_score"] == pytest.approx(0.1)
    assert summary["max_lrs_window_score"] == pytest.approx(0.1)


def test_rollout_summary_all_nan_is_nan():
    summary = _rollout_summary([float("nan"), float("nan")], [float("nan"), float("nan")])
    assert math.isnan(summary["mean_repetition_score"])
    assert math.isnan(summary["max_repetition_score"])
    assert math.isnan(summary["mean_lrs_window_score"])
    assert math.isnan(summary["max_lrs_window_score"])


def _labels_row(prompt_id, rollout_idx, repetition, lrs, entropy=None):
    n = len(repetition)
    return {
        "prompt_id": prompt_id,
        "rollout_idx": rollout_idx,
        "repetition_score": repetition,
        "lrs_window_score": lrs,
        "entropy": entropy if entropy is not None else [0.0] * n,
    }


def test_aggregate_prompt_stats_hand_computed():
    # p0: 2 rollouts.
    #   rollout0: repetition=[0.9, nan] -> mean=max=0.9 (> 0.8 threshold: degenerating)
    #             lrs=[0.1, nan] -> mean=max=0.1
    #   rollout1: repetition=[0.5, 0.5] -> mean=max=0.5 (<= 0.8: not degenerating)
    #             lrs=[0.2, 0.3] -> mean=0.25, max=0.3
    #   -> mean_repetition = mean(0.9, 0.5) = 0.7 ; max_repetition = max(0.9, 0.5) = 0.9
    #   -> mean_lrs = mean(0.1, 0.25) = 0.175 ; max_lrs = max(0.1, 0.3) = 0.3
    #   -> degeneration_rate = 1/2 = 0.5
    # p1: 1 rollout: repetition=[0.95] (> 0.8: degenerating), lrs=[0.4]
    #   -> mean_repetition=max_repetition=0.95 ; mean_lrs=max_lrs=0.4 ; degeneration_rate=1.0
    labels_df = pd.DataFrame.from_records(
        [
            _labels_row("p0", 0, [0.9, float("nan")], [0.1, float("nan")]),
            _labels_row("p0", 1, [0.5, 0.5], [0.2, 0.3]),
            _labels_row("p1", 0, [0.95], [0.4]),
        ]
    )
    prompt_id_to_domain = {"p0": "d0", "p1": "d1"}

    stats = aggregate_prompt_stats(labels_df, prompt_id_to_domain, degeneration_threshold=0.8)
    stats = stats.set_index("prompt_id")

    assert stats.loc["p0", "domain"] == "d0"
    assert stats.loc["p0", "n_rollouts"] == 2
    assert stats.loc["p0", "mean_repetition_score"] == pytest.approx(0.7)
    assert stats.loc["p0", "max_repetition_score"] == pytest.approx(0.9)
    assert stats.loc["p0", "mean_lrs_window_score"] == pytest.approx(0.175)
    assert stats.loc["p0", "max_lrs_window_score"] == pytest.approx(0.3)
    assert stats.loc["p0", "degeneration_rate"] == pytest.approx(0.5)

    assert stats.loc["p1", "domain"] == "d1"
    assert stats.loc["p1", "n_rollouts"] == 1
    assert stats.loc["p1", "mean_repetition_score"] == pytest.approx(0.95)
    assert stats.loc["p1", "max_repetition_score"] == pytest.approx(0.95)
    assert stats.loc["p1", "mean_lrs_window_score"] == pytest.approx(0.4)
    assert stats.loc["p1", "max_lrs_window_score"] == pytest.approx(0.4)
    assert stats.loc["p1", "degeneration_rate"] == pytest.approx(1.0)


def test_aggregate_prompt_stats_rollout_with_no_valid_windows_does_not_count_as_degenerating():
    labels_df = pd.DataFrame.from_records(
        [
            _labels_row("p0", 0, [float("nan"), float("nan")], [float("nan"), float("nan")]),
        ]
    )
    stats = aggregate_prompt_stats(labels_df, {"p0": "d0"}, degeneration_threshold=0.8)
    row = stats.iloc[0]
    assert row["n_rollouts"] == 1
    assert row["degeneration_rate"] == 0.0
    assert math.isnan(row["mean_repetition_score"])
    assert math.isnan(row["max_repetition_score"])


def test_aggregate_prompt_stats_covers_every_prompt_exactly_once():
    rows = []
    for i in range(5):
        pid = f"p{i}"
        for r in range(3):
            rows.append(_labels_row(pid, r, [0.1, 0.2], [0.0, 0.0]))
    labels_df = pd.DataFrame.from_records(rows)
    prompt_id_to_domain = {f"p{i}": "d0" for i in range(5)}

    stats = aggregate_prompt_stats(labels_df, prompt_id_to_domain)
    assert len(stats) == 5
    assert set(stats["prompt_id"]) == {f"p{i}" for i in range(5)}
    assert (stats["n_rollouts"] == 3).all()
