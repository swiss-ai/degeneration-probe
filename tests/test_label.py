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
find_longest_repeated_substring = label_module.find_longest_repeated_substring
_minimal_period = label_module._minimal_period
_merge_spans = label_module._merge_spans
_scan_for_unit_occurrences = label_module._scan_for_unit_occurrences
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


# --- find_longest_repeated_substring (whole-rollout, exact match) --------------

def test_find_longest_repeated_substring_obvious_repeated_block():
    # tokens = [1,2,3,1,2,3,7,8], min_length=2. The block "1,2,3" at index 0
    # matches the one at index 3: length 3, non-overlapping (3-0=3 >= 3). No
    # longer match exists (n//2=4, and no length-4 window repeats).
    tokens = [1, 2, 3, 1, 2, 3, 7, 8]
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 3
    assert result["lrs_score"] == pytest.approx(3 / 8)
    assert result["lrs_first_start"] == 0
    assert result["lrs_second_start"] == 3
    assert result["lrs_gap"] == 3
    assert result["lrs_repeated_token_ids"] == [1, 2, 3]


def test_find_longest_repeated_substring_no_repeats_returns_no_match():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 0
    assert result["lrs_score"] == 0.0
    assert result["lrs_first_start"] is None
    assert result["lrs_second_start"] is None
    assert result["lrs_gap"] is None
    assert result["lrs_repeated_token_ids"] == []


def test_find_longest_repeated_substring_min_length_gates_short_repeats():
    # The only repeat is "1" at index 0 and 1 (length 1), shorter than
    # min_length=3, so no length-3+ window repeats anywhere -> no match,
    # even though n (6) >= 2*min_length (6) so the search does run.
    tokens = [1, 1, 2, 3, 4, 5]
    result = find_longest_repeated_substring(tokens, min_length=3)
    assert result["lrs_length"] == 0
    assert result["lrs_repeated_token_ids"] == []


def test_find_longest_repeated_substring_uniformly_repetitive_sequence():
    # 8 identical tokens, min_length=1, n//2=4. Every length-4 window is
    # identical, so the first non-overlapping pair found scanning left to
    # right is (0, 4): 4-0=4 >= 4. No length beyond 4 is searchable (n//2=4).
    tokens = [9] * 8
    result = find_longest_repeated_substring(tokens, min_length=1)
    assert result["lrs_length"] == 4
    assert result["lrs_score"] == pytest.approx(0.5)
    assert result["lrs_first_start"] == 0
    assert result["lrs_second_start"] == 4
    assert result["lrs_gap"] == 4
    assert result["lrs_repeated_token_ids"] == [9, 9, 9, 9]


def test_find_longest_repeated_substring_finds_match_anywhere_not_just_adjacent():
    # A 5-token chunk repeated far apart, with distinct filler everywhere
    # else -- checks the search isn't restricted to nearby/adjacent
    # positions. Chunk occupies indices [3:8) and again [14:19); everything
    # else is a distinct token (1-11), so no other/accidental repeat exists
    # and no extension past the chunk boundary is possible (neighbors differ
    # on both sides).
    chunk = [101, 102, 103, 104, 105]
    tokens = [1, 2, 3] + chunk + [4, 5, 6, 7, 8, 9] + chunk + [10, 11]
    result = find_longest_repeated_substring(tokens, min_length=3)
    assert result["lrs_length"] == 5
    assert result["lrs_first_start"] == 3
    assert result["lrs_second_start"] == 14
    assert result["lrs_gap"] == 11
    assert result["lrs_repeated_token_ids"] == chunk
    assert result["lrs_score"] == pytest.approx(5 / len(tokens))


def test_find_longest_repeated_substring_close_repeat_can_still_beat_a_longer_gated_one():
    # tokens = [1,1,1,1,2,3,4,5]: only repeated value is "1" (positions 0-3).
    # L=2: window [1,1] recurs at starts 0 and 2 (2-0=2 >= 2, non-overlapping)
    #   -> succeeds. L=3: window [1,1,1] only recurs at starts 0 and 1, but
    #   1-0=1 < 3 (would overlap), and no other length-3 window repeats -- so
    #   L=3 fails even though L=2 succeeded (monotonicity only guarantees
    #   success at L implies success at every *shorter* length, not the
    #   reverse), so the binary search correctly settles on the true max, 2.
    tokens = [1, 1, 1, 1, 2, 3, 4, 5]
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 2
    assert result["lrs_score"] == pytest.approx(2 / 8)
    assert result["lrs_first_start"] == 0
    assert result["lrs_second_start"] == 2
    assert result["lrs_gap"] == 2
    assert result["lrs_repeated_token_ids"] == [1, 1]


# --- _minimal_period / occurrence scan ------------------------------------------

def test_minimal_period_finds_repeating_unit():
    # [1,2,3,1,2,3]: period-3 (m=6, evenly divisible).
    assert _minimal_period([1, 2, 3, 1, 2, 3]) == 3


def test_minimal_period_all_same_token_is_one():
    assert _minimal_period([9, 9, 9, 9]) == 1


def test_minimal_period_no_internal_repetition_is_full_length():
    # All-distinct tokens: no smaller period exists, so p = m itself.
    assert _minimal_period([1, 2, 3, 4]) == 4


def test_minimal_period_holds_even_when_it_does_not_evenly_divide_length():
    # "1,2,1,2,1" (m=5): tokens[i] == tokens[i+2] holds for i=0,1,2
    # (1==1, 2==2, 1==1) even though 5 isn't a multiple of 2.
    assert _minimal_period([1, 2, 1, 2, 1]) == 2


def test_occurrence_scan_merges_a_full_periodic_run():
    # period-3 unit [1,2,3] repeated exactly 4 times, filling the sequence.
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert _scan_for_unit_occurrences(tokens, unit_start=0, period=3) == [(0, 12)]


def test_occurrence_scan_stops_at_a_break():
    # period-2 unit [1,2] repeated 3 times, then something unrelated.
    tokens = [1, 2, 1, 2, 1, 2, 9, 9, 9]
    assert _scan_for_unit_occurrences(tokens, unit_start=0, period=2) == [(0, 6)]


# --- _merge_spans / occurrence scan ---------------------------------------------

def test_merge_spans_combines_overlapping_and_touching():
    assert _merge_spans([(0, 5), (5, 10), (20, 25)]) == [(0, 10), (20, 25)]


def test_merge_spans_keeps_disjoint_spans_separate():
    assert _merge_spans([(8, 13), (0, 5)]) == [(0, 5), (8, 13)]


def test_occurrence_scan_merges_when_gap_is_a_multiple_of_period():
    # [1,2,3] repeated 4 times (gap between occurrences 0 and 6 is a multiple
    # of period=3) -- extending from either occurrence reaches the same
    # single contiguous run.
    tokens = [1, 2, 3] * 4
    regions = _scan_for_unit_occurrences(tokens, unit_start=0, period=3)
    assert regions == [(0, 12)]


def test_occurrence_scan_keeps_disjoint_when_gap_is_unrelated_content():
    # An atomic (period == length) 5-token chunk repeated at positions 0 and
    # 8, with 3 tokens of unrelated filler in between (gap=8 is not a
    # multiple of period=5) -- the two occurrences must stay two separate
    # spans, not collapse into (or silently lose) one of them.
    paragraph = [10, 20, 30, 40, 50]
    filler = [99, 98, 97]
    tokens = paragraph + filler + paragraph
    regions = _scan_for_unit_occurrences(tokens, unit_start=0, period=5)
    assert regions == [(0, 5), (8, 13)]


# --- find_longest_repeated_substring: period decomposition ----------------------

def test_find_longest_repeated_substring_decomposes_short_unit_repeated_many_times():
    # A 3-token unit [1,2,3] repeated 4 times fills the whole 12-token
    # sequence. The core algorithm finds ONE pair (the biggest non-overlapping
    # match it can, by construction n//2=6 tokens long: start=0 vs start=6,
    # both "[1,2,3,1,2,3]"), but the period decomposition recovers the true
    # 3-token repeating unit and that it spans the whole sequence 4 times --
    # exactly the "short phrase spammed many times" signature.
    tokens = [1, 2, 3] * 4
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 6
    assert result["lrs_first_start"] == 0
    assert result["lrs_second_start"] == 6
    assert result["lrs_period"] == 3
    assert result["lrs_region_starts"] == [0]
    assert result["lrs_region_ends"] == [12]
    assert result["lrs_period_repeat_count"] == 4
    assert result["lrs_unit_token_ids"] == [1, 2, 3]


def test_find_longest_repeated_substring_does_not_decompose_a_twice_repeated_paragraph():
    # A 5-token "paragraph" with no internal repetition, repeated exactly
    # twice back-to-back. The period decomposition should NOT find a smaller
    # unit inside it -- lrs_period == lrs_length, repeat_count == 2 -- unlike
    # the short-unit-spammed case above, distinguishing "one big chunk said
    # twice" from "one small phrase spammed many times".
    paragraph = [10, 20, 30, 40, 50]
    tokens = paragraph + paragraph
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 5
    assert result["lrs_period"] == 5
    assert result["lrs_region_starts"] == [0]
    assert result["lrs_region_ends"] == [10]
    assert result["lrs_period_repeat_count"] == 2
    assert result["lrs_unit_token_ids"] == paragraph


def test_find_longest_repeated_substring_atomic_chunk_repeated_with_a_real_gap():
    # Regression test: an atomic (no internal period) 5-token chunk repeated
    # at positions 0 and 8, with unrelated filler in between (gap=8 is not a
    # multiple of period=5, i.e. this is NOT one contiguous periodic run).
    # Bug this guards against: seeding the region extension from only
    # first_start silently lost the second occurrence entirely whenever the
    # gap wasn't itself periodic, producing a nonsensical
    # "repeated 1x"/single-span result even though the algorithm had
    # correctly found two real occurrences (lrs_first_start/lrs_second_start
    # were always right -- only the period-decomposition fields were wrong).
    paragraph = [10, 20, 30, 40, 50]
    filler = [99, 98, 97]
    tokens = paragraph + filler + paragraph
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 5
    assert result["lrs_first_start"] == 0
    assert result["lrs_second_start"] == 8
    assert result["lrs_gap"] == 8
    assert result["lrs_period"] == 5
    assert result["lrs_region_starts"] == [0, 8]
    assert result["lrs_region_ends"] == [5, 13]
    assert result["lrs_period_repeat_count"] == 2
    assert result["lrs_unit_token_ids"] == paragraph


def test_find_longest_repeated_substring_keeps_the_exact_matched_unit():
    # The exact LRS match includes the marker shared by the first two cycles.
    # The occurrence scan does not trim a suffix merely to find more cycles.
    unit = [10, 20, 30, 40, 50]

    def cycle(i, marker):
        return unit + [marker] + [900 + i, 800 + i]

    markers = [77, 77, 88, 99, 111, 222]
    tokens = []
    for i, marker in enumerate(markers):
        tokens += cycle(i, marker)

    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_length"] == 6  # unit + the accidentally-shared marker
    assert result["lrs_first_start"] == 0
    assert result["lrs_second_start"] == 8
    assert result["lrs_period"] == 6
    assert result["lrs_unit_token_ids"] == unit + [77]
    assert result["lrs_region_starts"] == [0, 8]
    assert result["lrs_region_ends"] == [6, 14]
    assert result["lrs_period_repeat_count"] == 2


def test_find_longest_repeated_substring_no_match_has_none_period_fields():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    result = find_longest_repeated_substring(tokens, min_length=2)
    assert result["lrs_period"] is None
    assert result["lrs_period_repeat_count"] is None
    assert result["lrs_region_starts"] == []
    assert result["lrs_region_ends"] == []
    assert result["lrs_unit_token_ids"] == []


# --- label_rollout / label_shard ------------------------------------------------

def test_label_rollout_shapes_and_entropy_passthrough():
    token_ids = [1, 1, 1, 1, 2, 3, 4, 5]
    entropy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    labels = label_rollout(token_ids, entropy, window_size=4, ttr_ngram=1, lrs_min_length=2)

    assert len(labels["repetition_score"]) == len(token_ids)
    assert len(labels["entropy"]) == len(token_ids)
    assert labels["entropy"] == pytest.approx(entropy)
    # Same hand-computed ttr as test_sliding_window_repetition_hand_computed.
    assert labels["repetition_score"][0] == pytest.approx(0.75)
    assert labels["repetition_score"][3] == pytest.approx(0.0)
    assert math.isnan(labels["repetition_score"][-1])
    # Same hand-computed LRS match as
    # test_find_longest_repeated_substring_close_repeat_can_still_beat_a_longer_gated_one.
    assert labels["lrs_length"] == 2
    assert labels["lrs_first_start"] == 0
    assert labels["lrs_second_start"] == 2
    assert labels["lrs_repeated_token_ids"] == [1, 1]


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

    assert list(labels_df.columns) == label_module.LABEL_COLUMNS
    assert len(labels_df) == 2
    assert set(labels_df["rollout_idx"]) == {0, 1}

    row0 = labels_df[labels_df["rollout_idx"] == 0].iloc[0]
    assert row0["repetition_score"][0] == pytest.approx(0.75)
    assert row0["entropy"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # Repeated "1,1" block, as in test_label_rollout_shapes_and_entropy_passthrough.
    assert row0["lrs_length"] == 2
    assert row0["lrs_repeated_token_ids"] == [1, 1]

    row1 = labels_df[labels_df["rollout_idx"] == 1].iloc[0]
    # All-distinct tokens -> repetition score 0.0 everywhere a window fits, no LRS match.
    assert row1["repetition_score"][0] == pytest.approx(0.0)
    assert row1["lrs_length"] == 0
    assert row1["lrs_repeated_token_ids"] == []


# --- _rollout_summary / aggregate_prompt_stats ---------------------------------

def test_rollout_summary_ignores_nan_tail():
    repetition = [0.9, float("nan")]
    summary = _rollout_summary(repetition)
    assert summary["mean_repetition_score"] == pytest.approx(0.9)
    assert summary["max_repetition_score"] == pytest.approx(0.9)


def _labels_row(prompt_id, rollout_idx, repetition, lrs_score, entropy=None):
    n = len(repetition)
    return {
        "prompt_id": prompt_id,
        "rollout_idx": rollout_idx,
        "repetition_score": repetition,
        "lrs_score": lrs_score,
        "entropy": entropy if entropy is not None else [0.0] * n,
    }


def test_aggregate_prompt_stats_hand_computed():
    # p0: 2 rollouts.
    #   rollout0: repetition=[0.9, nan] -> mean=max=0.9 (> 0.8 threshold: degenerating); lrs_score=0.1
    #   rollout1: repetition=[0.5, 0.5] -> mean=max=0.5 (<= 0.8: not degenerating); lrs_score=0.3
    #   -> mean_repetition = mean(0.9, 0.5) = 0.7 ; max_repetition = max(0.9, 0.5) = 0.9
    #   -> mean_lrs = mean(0.1, 0.3) = 0.2 ; max_lrs = max(0.1, 0.3) = 0.3
    #   -> degeneration_rate = 1/2 = 0.5
    # p1: 1 rollout: repetition=[0.95] (> 0.8: degenerating), lrs_score=0.4
    #   -> mean_repetition=max_repetition=0.95 ; mean_lrs=max_lrs=0.4 ; degeneration_rate=1.0
    labels_df = pd.DataFrame.from_records(
        [
            _labels_row("p0", 0, [0.9, float("nan")], 0.1),
            _labels_row("p0", 1, [0.5, 0.5], 0.3),
            _labels_row("p1", 0, [0.95], 0.4),
        ]
    )
    prompt_id_to_domain = {"p0": "d0", "p1": "d1"}

    stats = aggregate_prompt_stats(labels_df, prompt_id_to_domain, degeneration_threshold=0.8)
    stats = stats.set_index("prompt_id")

    assert stats.loc["p0", "domain"] == "d0"
    assert stats.loc["p0", "n_rollouts"] == 2
    assert stats.loc["p0", "mean_repetition_score"] == pytest.approx(0.7)
    assert stats.loc["p0", "max_repetition_score"] == pytest.approx(0.9)
    assert stats.loc["p0", "mean_lrs_score"] == pytest.approx(0.2)
    assert stats.loc["p0", "max_lrs_score"] == pytest.approx(0.3)
    assert stats.loc["p0", "degeneration_rate"] == pytest.approx(0.5)

    assert stats.loc["p1", "domain"] == "d1"
    assert stats.loc["p1", "n_rollouts"] == 1
    assert stats.loc["p1", "mean_repetition_score"] == pytest.approx(0.95)
    assert stats.loc["p1", "max_repetition_score"] == pytest.approx(0.95)
    assert stats.loc["p1", "mean_lrs_score"] == pytest.approx(0.4)
    assert stats.loc["p1", "max_lrs_score"] == pytest.approx(0.4)
    assert stats.loc["p1", "degeneration_rate"] == pytest.approx(1.0)


def test_aggregate_prompt_stats_rollout_with_no_valid_windows_does_not_count_as_degenerating():
    labels_df = pd.DataFrame.from_records(
        [
            _labels_row("p0", 0, [float("nan"), float("nan")], 0.0),
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
            rows.append(_labels_row(pid, r, [0.1, 0.2], 0.0))
    labels_df = pd.DataFrame.from_records(rows)
    prompt_id_to_domain = {f"p{i}": "d0" for i in range(5)}

    stats = aggregate_prompt_stats(labels_df, prompt_id_to_domain)
    assert len(stats) == 5
    assert set(stats["prompt_id"]) == {f"p{i}" for i in range(5)}
    assert (stats["n_rollouts"] == 3).all()
