"""Phase 2 of the dataset generation pipeline: degeneration labels.

For every rollout in ``paths.generations_shard_path(config, domain, 0)`` (written
by ``generate.py``) this module computes:

1. ``repetition_score`` -- per-token, sliding-window ``1 - TTR`` (type-token
   ratio) over bigrams (n=2): for each position ``t``,
   ``1 - (distinct 2-grams in tokens[t:t+window_size]) / (window_size - 1)``.
   Positions where a full window doesn't fit at the end of the sequence are
   NaN (masked). A rollout counts as degenerate when this exceeds
   ``DEFAULT_DEGENERATION_THRESHOLD`` (0.8).

2. ``entropy`` -- per-token pass-through of Phase 1's ``per_token_entropy``
   column (already per-token, already aligned to ``generated_token_ids``).

3. LRS (Longest Repeated Substring) -- a *whole-rollout*, exact-match signal,
   computed once per rollout (not per token): the longest exact,
   non-overlapping substring that occurs twice anywhere in the rollout's
   tokens. Unlike (1)/(2) this isn't an array aligned to token positions --
   it's a single match (or "no match"), reported as several columns:
   ``lrs_length`` (tokens), ``lrs_score`` (``lrs_length / num_tokens``),
   ``lrs_first_start`` / ``lrs_second_start`` (the two occurrences' start
   indices), ``lrs_gap`` (``lrs_second_start - lrs_first_start`` -- small means
   the model repeats itself immediately, large means it echoes something said
   much earlier), and ``lrs_repeated_token_ids`` (the actual repeated chunk,
   for decoding/inspection). See ``find_longest_repeated_substring`` for the
   algorithm (binary search over candidate length + rolling hash, O(n log n),
   replacing an earlier per-token sliding-window LRS approximation that didn't
   scale to window_size=256).

Output: ``labels/<domain>/shard_00000.parquet`` (via
``paths.labels_shard_path``), one row per rollout, columns per
``LABEL_COLUMNS``.

This module also aggregates across ALL domains into
``prompt_stats/prompt_stats.parquet`` (via ``paths.prompt_stats_path``): one
row per prompt_id summarizing its ``n_rollouts_per_prompt`` (10) rollouts --
see ``aggregate_prompt_stats`` for exactly how "mean"/"max"/"degeneration_rate"
are defined across the two levels (per-rollout, then per-prompt).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "builds" / "degeneration-dataset-apertus-8b-instruct.yaml"

DEFAULT_WINDOW_SIZE = 256
DEFAULT_TTR_NGRAM = 2
DEFAULT_LRS_MIN_LENGTH = 10
# Threshold retained for rollout-summary diagnostics.
DEFAULT_DEGENERATION_THRESHOLD = 0.8

LABEL_COLUMNS = [
    "prompt_id",
    "rollout_idx",
    "repetition_score",
    "entropy",
    "lrs_length",
    "lrs_score",
    "lrs_first_start",
    "lrs_second_start",
    "lrs_gap",
    "lrs_repeated_token_ids",
    "lrs_period",
    "lrs_period_repeat_count",
    "lrs_region_starts",
    "lrs_region_ends",
    "lrs_unit_token_ids",
    # Digit-normalized LRS: same fields as the lrs_* block above, but with
    # runs of digit tokens ('0'-'9') collapsed to one wildcard unit before
    # matching, so a repeated template survives both a changed digit value
    # (e.g. "N = 76" vs "N = 77") and a changed digit *count* (e.g. "6" vs
    # "600") without breaking the match -- see
    # `find_longest_repeated_substring_growing_normalized`'s docstring. This
    # is the current best onset-position signal (`lrs_first_start_normalized_growing`);
    # validated against the LLM judge's `truncated` stratum to substantially
    # reduce onset lag vs. the plain lrs_* fields above, which are kept
    # unmodified since other code (`aggregate_prompt_stats`) depends on them.
    # An intermediate digit-*value*-only variant (no run-collapsing) was also
    # tried and dropped -- it fixed the value-changing case but not the
    # count-changing one, and this variant dominates it on every rollout
    # checked so far.
    "lrs_length_normalized_growing",
    "lrs_score_normalized_growing",
    "lrs_first_start_normalized_growing",
    "lrs_second_start_normalized_growing",
    "lrs_gap_normalized_growing",
    "lrs_repeated_token_ids_normalized_growing",
    "lrs_period_normalized_growing",
    "lrs_period_repeat_count_normalized_growing",
    "lrs_region_starts_normalized_growing",
    "lrs_region_ends_normalized_growing",
    "lrs_unit_token_ids_normalized_growing",
]

PROMPT_STATS_COLUMNS = [
    "prompt_id",
    "domain",
    "n_rollouts",
    "mean_repetition_score",
    "max_repetition_score",
    "mean_lrs_score",
    "max_lrs_score",
    "degeneration_rate",
]


# --- TTR window score -----------------------------------------------------------

def _ngram_ttr(token_ids: Sequence[int], n: int) -> float:
    """Type-token ratio over n-grams. 1.0 if too few tokens."""
    if len(token_ids) < n or n < 1:
        return 1.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def sliding_window_repetition(
    token_ids: Sequence[int],
    window_size: int,
    n: int = DEFAULT_TTR_NGRAM,
) -> List[float]:
    """
    For each position t in `token_ids`, return 1 - TTR computed over the
    window `token_ids[t : t + window_size]`. Positions where the full window
    does not fit (i.e. t + window_size > len) return NaN -- the caller is
    responsible for masking these out of the loss.
    """
    out: List[float] = []
    total = len(token_ids)
    for t in range(total):
        end = t + window_size
        if end > total:
            out.append(float("nan"))
        else:
            out.append(1.0 - _ngram_ttr(token_ids[t:end], n))
    return out


# --- LRS (longest repeated substring): exact, whole-rollout -------------------

# Polynomial-hash constants for the rolling hash used by
# `find_longest_repeated_substring`. Any collision is only ever treated as a
# *candidate* and verified with a direct token comparison before being
# trusted (see `_find_matching_pair`), so these don't need to be
# cryptographic -- just large enough that collisions are rare in practice.
_HASH_MOD = (1 << 61) - 1
_HASH_BASE = 131_542_391


def _no_lrs_match(key_suffix: str = "") -> Dict[str, object]:
    return {
        f"lrs_length{key_suffix}": 0,
        f"lrs_score{key_suffix}": 0.0,
        f"lrs_first_start{key_suffix}": None,
        f"lrs_second_start{key_suffix}": None,
        f"lrs_gap{key_suffix}": None,
        f"lrs_repeated_token_ids{key_suffix}": [],
        f"lrs_period{key_suffix}": None,
        f"lrs_period_repeat_count{key_suffix}": None,
        f"lrs_region_starts{key_suffix}": [],
        f"lrs_region_ends{key_suffix}": [],
        f"lrs_unit_token_ids{key_suffix}": [],
    }


# Digit tokens ('0'-'9') collapsed to this before matching, for the
# digit-run-normalized LRS variant -- see `digit_run_collapsed`. Never a real
# token id (all real ids are >= 0), so it can't collide.
_DIGIT_WILDCARD = -1


def digit_token_ids(tokenizer: AutoTokenizer) -> Set[int]:
    """Every vocab entry that's a bare digit ('0'-'9'), across any tokenizer-specific
    prefix marker (e.g. BPE's 'Ġ', SentencePiece's '▁'). Apertus's tokenizer only
    ever emits numbers as individual single-digit tokens (no merged multi-digit or
    space-fused variants), so this is small (10 ids) and exact for it -- a tokenizer
    with merged multi-digit tokens would still be handled correctly by the same
    `.isdigit()` check, just with a larger set."""
    return {i for s, i in tokenizer.get_vocab().items() if s.strip("Ġ▁").isdigit()}


def digit_run_collapsed(
    tokens: Sequence[int], digit_token_ids: Set[int]
) -> Tuple[List[int], List[int], List[int]]:
    """Collapse each maximal run of consecutive digit tokens into a single wildcard
    token, for the comparison-only "growing-numeral" LRS variant.

    `digit_normalized_tokens` substitutes one wildcard per digit token, which keeps
    the sequence length unchanged -- but that only fixes a template whose numeral has
    the same digit *count* every time (e.g. "76" vs "77"). It can't fix a numeral
    whose digit count itself grows between repeats (e.g. "6" then "60" then "600"),
    because an exact-match algorithm compares fixed-length windows: once two
    occurrences' numerals have different lengths, everything after the numeral in a
    fixed-length window is shifted relative to the other occurrence and stops lining
    up, no matter what the digits themselves are replaced with. Collapsing the whole
    run to one token fixes this by realigning positions regardless of how many digits
    each occurrence's numeral actually had.

    Returns `(compressed_tokens, run_start, run_length)`, three parallel lists over
    the *compressed* sequence: `compressed_tokens[i]` is that position's token (or
    the wildcard), `run_start[i]` is its start index back in the original `tokens`,
    and `run_length[i]` is how many original tokens it spans (1 normally, the full
    run length when it's a collapsed digit run) -- enough to map any compressed
    position or span back to real token positions.
    """
    compressed: List[int] = []
    run_start: List[int] = []
    run_length: List[int] = []
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i] in digit_token_ids:
            j = i
            while j < n and tokens[j] in digit_token_ids:
                j += 1
            compressed.append(_DIGIT_WILDCARD)
            run_start.append(i)
            run_length.append(j - i)
            i = j
        else:
            compressed.append(tokens[i])
            run_start.append(i)
            run_length.append(1)
            i += 1
    return compressed, run_start, run_length


def _minimal_period(tokens: Sequence[int]) -> int:
    """Smallest p (1 <= p <= len(tokens)) with `tokens[i] == tokens[i + p]` for
    every valid i.

    Standard string-periodicity result, via the KMP prefix function: for a
    length-m string, `p = m - pi[m-1]` is always its smallest period -- this
    holds whether or not p evenly divides m (e.g. "1,2,1,2,1" has period 2
    even though 5 isn't a multiple of 2). If the string has no internal
    repetition at all, this returns m itself (its only "period" is its own
    full length).
    """
    m = len(tokens)
    if m <= 1:
        return max(m, 1)
    pi = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and tokens[i] != tokens[k]:
            k = pi[k - 1]
        if tokens[i] == tokens[k]:
            k += 1
        pi[i] = k
    return m - pi[-1]


def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/touching (start, end) spans into a sorted, disjoint list."""
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# A repeat's recurrence interval isn't necessarily constant: if the
# "boilerplate" between one occurrence and the next varies in token length
# from cycle to cycle (e.g. two different labels that happen to tokenize to
# different lengths, as with math problems repeating a formula for
# differently-named parts), checking only exact multiples of one measured gap
# -- or only walking adjacent period-steps -- can step straight past a real
# occurrence and find nothing there. `_scan_for_unit_occurrences` instead
# checks *every* position in the sequence directly against the known
# repeating unit, so it finds every genuine occurrence regardless of how it's
# spaced relative to the others.


def _scan_for_unit_occurrences(
    tokens: Sequence[int],
    unit_start: int,
    period: int,
) -> List[Tuple[int, int]]:
    """Find every exact occurrence of the `period`-token unit
    `tokens[unit_start : unit_start + period]` anywhere in `tokens`, by
    directly scoring every candidate window rather than assuming occurrences
    are evenly spaced.

    Each candidate start `c` is scored by how many of its `period` tokens
    equal the unit position-for-position (not just a common prefix, since the
    part that differs between occurrences -- a differently-tokenizing label,
    say -- isn't necessarily at the end of the window), and only accepted as
    an occurrence if *all* `period` tokens match -- kept exact, like the
    initial match in `find_longest_repeated_substring`, rather than tolerating
    partial mismatches, so a rollout's repeat count never counts something
    LRS wouldn't also accept as a match on its own. Accepted windows are
    merged into disjoint spans, so a run of directly-adjacent occurrences (a
    continuous loop) collapses into one span the same way separated
    occurrences don't.

    Vectorized over `c` with numpy: for each of the `period` positions within
    the unit, one O(n) comparison finds every `c` where that position matches,
    and the per-`c` counts are accumulated across all `period` positions --
    O(n * period) total, but as `period` vectorized numpy ops rather than a
    Python-level nested loop.
    """
    n = len(tokens)
    arr = np.asarray(tokens)
    unit = arr[unit_start : unit_start + period]
    n_candidates = n - period + 1
    if n_candidates <= 0:
        return []

    match_counts = np.zeros(n_candidates, dtype=np.int32)
    for k in range(period):
        match_counts += (arr[k : k + n_candidates] == unit[k])

    accepted = np.flatnonzero(match_counts == period)
    spans = [(int(c), int(c) + period) for c in accepted]
    return _merge_spans(spans)


def find_longest_repeated_substring(
    token_ids: Sequence[int],
    min_length: int = DEFAULT_LRS_MIN_LENGTH,
    key_suffix: str = "",
) -> Dict[str, object]:
    """Find the longest exact, non-overlapping repeated substring in `token_ids`.

    Whole-rollout (not per-token) signal: binary search over the candidate
    match length L. "Does a non-overlapping repeat of length >= L exist
    anywhere in the sequence?" is monotonic in L (any length-L repeat's
    prefixes are themselves shorter repeats at the same two positions), so
    binary search finds the true maximum in O(log n) probes. Each probe is
    O(n): hash every length-L window with a rolling polynomial hash
    (Rabin-Karp), bucket start positions by hash, and accept a pair (p, i)
    from within a bucket only if `i - p >= L` (non-overlapping) and the
    tokens actually match -- a hash collision is just a candidate, always
    verified directly. Overall O(n log n) per rollout, independent of L, vs.
    the O(window_size^2)-ish cost *per token position* of the earlier
    sliding-window approach.

    `min_length` is the shortest match worth reporting at all (the binary
    search's lower bound) -- if the true longest repeat is shorter than
    this, it's treated as "no match" rather than returned.

    Also decomposes the match into its true repeating unit via
    `_minimal_period` / `_scan_for_unit_occurrences`: on fully-degenerate,
    gapless repetition this single (first_start, second_start) pair is
    really just two samples of one long periodic run -- e.g. a 1-token unit
    repeated 2000 times and a 2000-token paragraph repeated twice both
    produce `lrs_length` ~= n/2 with `lrs_gap` ~= `lrs_length`, and are only
    told apart by `lrs_period` (1 vs. ~2000). But the two occurrences aren't
    always part of one contiguous run -- a short, atomic phrase (`lrs_period
    == lrs_length`) can genuinely repeat twice with unrelated content in
    between (`lrs_gap` not a multiple of `lrs_period`) -- so the repeating
    unit can occupy more than one disjoint span; `lrs_region_starts` /
    `lrs_region_ends` are therefore parallel lists, not a single range.

    Once the `lrs_period`-token unit is known, `_scan_for_unit_occurrences`
    looks for it everywhere in the sequence (not just adjacent to the
    original pair, and not just at multiples of `lrs_gap`), scoring every
    candidate position directly rather than assuming occurrences are evenly
    spaced -- this matters for loops that splice in some varying element
    each cycle (a step counter, or a differently-tokenizing label), since the
    gap between one pair of occurrences isn't a reliable predictor of the
    gap to the next. `lrs_period_repeat_count` reflects the true number of
    occurrences found this way, which can exceed 2.

    Returns a dict:
    - `lrs_length`, `lrs_score` (`lrs_length / len(token_ids)`)
    - `lrs_first_start`, `lrs_second_start`, `lrs_gap`
      (`lrs_second_start - lrs_first_start`)
    - `lrs_repeated_token_ids` -- the raw matched chunk (length `lrs_length`)
    - `lrs_period` -- the matched chunk's own smallest internal period (in
      tokens); equals `lrs_length` itself if the chunk has no smaller
      internal repetition (e.g. a long paragraph repeated exactly twice)
    - `lrs_region_starts` / `lrs_region_ends` -- parallel lists of the
      disjoint span(s) (may reach beyond both `lrs_first_start` and
      `lrs_second_start + lrs_length`) over which that `lrs_period`-token
      unit repeats -- one span if it's one contiguous periodic run, two if
      the occurrences are genuinely separate
    - `lrs_period_repeat_count` -- total repeats of the unit summed across
      all spans: `sum((end - start) // lrs_period for start, end in spans)`
    - `lrs_unit_token_ids` -- the actual `lrs_period`-token repeating unit
      (from the first span), for decoding/inspection/highlighting every
      occurrence

    All zero/None/empty if no repeat of at least `min_length` tokens exists.

    `key_suffix` is appended to every key in the returned dict, so this can be
    called more than once (e.g. on a transformed token sequence, as
    `find_longest_repeated_substring_growing_normalized` does) and the results
    merged into one flat dict without colliding.
    """
    n = len(token_ids)
    tokens = list(token_ids)
    match = tokens

    if n < 2 * min_length:
        return _no_lrs_match(key_suffix)

    # Prefix hashes / powers are independent of L -- computed once, then any
    # window's hash is O(1): hash(match[i:i+L]) = prefix[i+L] - prefix[i] * power[L].
    prefix = [0] * (n + 1)
    power = [1] * (n + 1)
    for idx in range(n):
        prefix[idx + 1] = (prefix[idx] * _HASH_BASE + match[idx] + 1) % _HASH_MOD
        power[idx + 1] = (power[idx] * _HASH_BASE) % _HASH_MOD

    def window_hash(i: int, length: int) -> int:
        return (prefix[i + length] - prefix[i] * power[length]) % _HASH_MOD

    def find_matching_pair(length: int):
        """First verified non-overlapping (first_start, second_start) pair
        with an exact match of this length, or None if none exists."""
        buckets: Dict[int, List[int]] = {}
        for i in range(n - length + 1):
            h = window_hash(i, length)
            bucket = buckets.get(h)
            if bucket is None:
                buckets[h] = [i]
                continue
            for p in bucket:
                if i - p >= length and match[p : p + length] == match[i : i + length]:
                    return (p, i)
            bucket.append(i)
        return None

    lo, hi = min_length, n // 2
    best_length = 0
    best_pair = None
    while lo <= hi:
        mid = (lo + hi) // 2
        pair = find_matching_pair(mid)
        if pair is not None:
            best_length, best_pair = mid, pair
            lo = mid + 1
        else:
            hi = mid - 1

    if best_pair is None:
        return _no_lrs_match(key_suffix)

    first_start, second_start = best_pair
    matched_span = tokens[first_start : first_start + best_length]
    period = _minimal_period(match[first_start : first_start + best_length])
    regions = _scan_for_unit_occurrences(match, first_start, period)
    region_starts = [s for s, _e in regions]
    region_ends = [e for _s, e in regions]
    first_region_start = region_starts[0]

    return {
        f"lrs_length{key_suffix}": best_length,
        f"lrs_score{key_suffix}": best_length / n,
        f"lrs_first_start{key_suffix}": first_start,
        f"lrs_second_start{key_suffix}": second_start,
        f"lrs_gap{key_suffix}": second_start - first_start,
        f"lrs_repeated_token_ids{key_suffix}": matched_span,
        f"lrs_period{key_suffix}": period,
        f"lrs_period_repeat_count{key_suffix}": sum((e - s) // period for s, e in regions),
        f"lrs_region_starts{key_suffix}": region_starts,
        f"lrs_region_ends{key_suffix}": region_ends,
        f"lrs_unit_token_ids{key_suffix}": tokens[first_region_start : first_region_start + period],
    }


def find_longest_repeated_substring_growing_normalized(
    token_ids: Sequence[int],
    digit_token_ids: Set[int],
    min_length: int = DEFAULT_LRS_MIN_LENGTH,
    key_suffix: str = "_normalized_growing",
) -> Dict[str, object]:
    """Comparison-only LRS variant that also handles a growing-length numeral (e.g.
    "6" -> "60" -> "600" -- see `digit_run_collapsed`'s docstring for why the
    same-length `digit_normalized_tokens` substitution can't fix this case).

    Runs the same matching algorithm on `digit_run_collapsed(token_ids, ...)`'s
    compressed sequence (so positions realign across numerals of different digit
    counts), then maps every returned position back to real `token_ids` indices via
    that call's `run_start`/`run_length`.

    Unlike the plain and `_normalized` variants, this one's `lrs_length`/`lrs_period`
    are NOT directly comparable to theirs: a growing numeral means the *real* token
    length of "one repeat" differs between occurrences by construction, so there's no
    single real-token-count answer -- `lrs_length` here specifically reports the
    first occurrence's real length, and `lrs_period` is measured in compressed-token
    units (structural position count, not real token count). `lrs_first_start` /
    `lrs_second_start` / `lrs_region_starts` / `lrs_region_ends`, which is what onset
    localization actually depends on, are real token positions like the other two
    variants -- `lrs_period_repeat_count` is a plain count and also transfers as-is.
    """
    tokens = [int(t) for t in token_ids]
    compressed, run_start, run_length = digit_run_collapsed(tokens, digit_token_ids)
    raw = find_longest_repeated_substring(compressed, min_length=min_length)
    if raw["lrs_first_start"] is None:
        return _no_lrs_match(key_suffix)

    def real_start(comp_idx: int) -> int:
        return run_start[comp_idx]

    def real_end_exclusive(comp_end_exclusive: int) -> int:
        last = comp_end_exclusive - 1
        return run_start[last] + run_length[last]

    first_start = real_start(raw["lrs_first_start"])
    first_end = real_end_exclusive(raw["lrs_first_start"] + raw["lrs_length"])
    second_start = real_start(raw["lrs_second_start"])

    region_starts = [real_start(s) for s in raw["lrs_region_starts"]]
    region_ends = [real_end_exclusive(e) for e in raw["lrs_region_ends"]]
    first_region_start = region_starts[0]
    # One period's real span, taken from its first occurrence in the first region.
    unit_comp_end = raw["lrs_region_starts"][0] + raw["lrs_period"]
    unit_real_end = real_end_exclusive(unit_comp_end)

    return {
        f"lrs_length{key_suffix}": first_end - first_start,
        f"lrs_score{key_suffix}": (first_end - first_start) / len(tokens) if tokens else 0.0,
        f"lrs_first_start{key_suffix}": first_start,
        f"lrs_second_start{key_suffix}": second_start,
        f"lrs_gap{key_suffix}": second_start - first_start,
        f"lrs_repeated_token_ids{key_suffix}": tokens[first_start:first_end],
        f"lrs_period{key_suffix}": raw["lrs_period"],
        f"lrs_period_repeat_count{key_suffix}": raw["lrs_period_repeat_count"],
        f"lrs_region_starts{key_suffix}": region_starts,
        f"lrs_region_ends{key_suffix}": region_ends,
        f"lrs_unit_token_ids{key_suffix}": tokens[first_region_start:unit_real_end],
    }


# --- per-rollout / per-shard labeling ------------------------------------------

def label_rollout(
    generated_token_ids: Sequence[int],
    per_token_entropy: Sequence[float],
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttr_ngram: int = DEFAULT_TTR_NGRAM,
    lrs_min_length: int = DEFAULT_LRS_MIN_LENGTH,
    digit_token_ids: Optional[Set[int]] = None,
) -> Dict[str, object]:
    """Compute the per-token repetition/entropy signals plus the whole-rollout LRS match.

    `digit_token_ids`, if given, additionally computes the digit-run-normalized
    LRS variant (`lrs_*_normalized_growing` -- see
    `find_longest_repeated_substring_growing_normalized`), currently the best
    available onset-position signal (see `LABEL_COLUMNS`'s comment). If
    omitted, those fields are filled with "no match" placeholders so the
    output schema is always complete.
    """
    token_ids = [int(t) for t in generated_token_ids]
    repetition = sliding_window_repetition(token_ids, window_size, n=ttr_ngram)
    entropy = [float(e) for e in per_token_entropy]
    lrs = find_longest_repeated_substring(token_ids, min_length=lrs_min_length)
    if digit_token_ids:
        lrs_normalized_growing = find_longest_repeated_substring_growing_normalized(
            token_ids, digit_token_ids, min_length=lrs_min_length,
        )
    else:
        lrs_normalized_growing = _no_lrs_match("_normalized_growing")
    return {
        "repetition_score": repetition,
        "entropy": entropy,
        **lrs,
        **lrs_normalized_growing,
    }


def _label_rollout_task(
    task: Tuple[str, int, Sequence[int], Sequence[float], int, int, int, Optional[Set[int]]]
) -> dict:
    """Picklable ProcessPoolExecutor worker: label one rollout, tagged with its keys."""
    (
        prompt_id,
        rollout_idx,
        generated_token_ids,
        per_token_entropy,
        window_size,
        ttr_ngram,
        lrs_min_length,
        digit_token_ids,
    ) = task
    labels = label_rollout(
        generated_token_ids,
        per_token_entropy,
        window_size=window_size,
        ttr_ngram=ttr_ngram,
        lrs_min_length=lrs_min_length,
        digit_token_ids=digit_token_ids,
    )
    return {"prompt_id": prompt_id, "rollout_idx": rollout_idx, **labels}


def label_shard(
    generations_df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttr_ngram: int = DEFAULT_TTR_NGRAM,
    lrs_min_length: int = DEFAULT_LRS_MIN_LENGTH,
    n_workers: int = 1,
    digit_token_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Label every rollout in one domain's generations shard.

    Each rollout's labeling is independent, so with ``n_workers > 1``
    rollouts are farmed out to a ``ProcessPoolExecutor``. ``executor.map``
    returns results in the same order as the input tasks (not completion
    order), so the returned DataFrame's row order always matches
    ``generations_df``'s.
    """
    tasks = [
        (
            row.prompt_id,
            int(row.rollout_idx),
            row.generated_token_ids,
            row.per_token_entropy,
            window_size,
            ttr_ngram,
            lrs_min_length,
            digit_token_ids,
        )
        for row in generations_df.itertuples(index=False)
    ]

    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            records = list(executor.map(_label_rollout_task, tasks, chunksize=1))
    else:
        records = [_label_rollout_task(t) for t in tasks]

    return pd.DataFrame.from_records(records, columns=LABEL_COLUMNS)


def write_shard_atomic(df: pd.DataFrame, shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(shard_path.parent), prefix=".tmp_shard_", suffix=".parquet"
    )
    os.fchmod(fd, 0o640)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, shard_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# --- prompt-level aggregation --------------------------------------------------

def _rollout_summary(repetition: Sequence[float]) -> Dict[str, float]:
    """Reduce one rollout's per-token repetition array to a (mean, max) pair.

    NaN-tail positions are ignored (np.nanmean/np.nanmax). If a rollout has no
    valid windows at all (shorter than window_size -- doesn't happen in
    practice, but handled defensively), both reduce to NaN rather than
    raising. (LRS needs no such reduction -- `find_longest_repeated_substring`
    already returns one `lrs_score` per rollout, not a per-token array.)
    """
    repetition_arr = np.asarray(repetition, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice
        mean_repetition = (
            float(np.nanmean(repetition_arr)) if np.any(~np.isnan(repetition_arr)) else float("nan")
        )
        max_repetition = (
            float(np.nanmax(repetition_arr)) if np.any(~np.isnan(repetition_arr)) else float("nan")
        )

    return {
        "mean_repetition_score": mean_repetition,
        "max_repetition_score": max_repetition,
    }


def aggregate_prompt_stats(
    labels_df: pd.DataFrame,
    prompt_id_to_domain: Dict[str, str],
    degeneration_threshold: float = DEFAULT_DEGENERATION_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate a (multi-domain) labels table to one row per prompt_id.

    For each rollout, first reduce its per-token repetition array to a (mean,
    max) pair (NaN-tail positions ignored); `lrs_score` is already a single
    number per rollout. Then, per prompt_id, across its rollouts:
      - mean_repetition_score / mean_lrs_score: mean of the per-rollout
        values (NaN rollouts, i.e. ones with no valid window, are ignored).
      - max_repetition_score / max_lrs_score: max of the per-rollout values.
      - degeneration_rate: fraction of rollouts whose max_repetition_score
        exceeds `degeneration_threshold` (a rollout with an undefined,
        NaN max never counts as exceeding it).
    """
    per_rollout_records = []
    for row in labels_df.itertuples(index=False):
        summary = _rollout_summary(row.repetition_score)
        summary["prompt_id"] = row.prompt_id
        summary["rollout_idx"] = int(row.rollout_idx)
        summary["lrs_score"] = float(row.lrs_score)
        per_rollout_records.append(summary)

    per_rollout_df = pd.DataFrame.from_records(per_rollout_records)

    prompt_records = []
    for prompt_id, group in per_rollout_df.groupby("prompt_id", sort=False):
        n_rollouts = len(group)
        max_repetition_vals = group["max_repetition_score"].to_numpy(dtype=float)
        # NaN comparisons are False, so a rollout with an undefined max never
        # counts toward the degeneration rate.
        n_degenerating = int(np.sum(max_repetition_vals > degeneration_threshold))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_repetition = float(group["mean_repetition_score"].mean(skipna=True))
            max_repetition = float(group["max_repetition_score"].max(skipna=True))
            mean_lrs = float(group["lrs_score"].mean(skipna=True))
            max_lrs = float(group["lrs_score"].max(skipna=True))

        prompt_records.append(
            {
                "prompt_id": prompt_id,
                "domain": prompt_id_to_domain.get(prompt_id),
                "n_rollouts": n_rollouts,
                "mean_repetition_score": mean_repetition,
                "max_repetition_score": max_repetition,
                "mean_lrs_score": mean_lrs,
                "max_lrs_score": max_lrs,
                "degeneration_rate": n_degenerating / n_rollouts,
            }
        )

    return pd.DataFrame.from_records(prompt_records, columns=PROMPT_STATS_COLUMNS)


# --- CLI entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to the DatasetGenConfig YAML file.",
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="Subset of domains to label (default: all configured domains).",
    )
    parser.add_argument(
        "--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
        help="Sliding-window size (in tokens) for the per-token repetition_score (default: 256).",
    )
    parser.add_argument("--ttr-ngram", type=int, default=DEFAULT_TTR_NGRAM)
    parser.add_argument(
        "--lrs-min-length", type=int, default=DEFAULT_LRS_MIN_LENGTH,
        help="Shortest exact repeated substring (in tokens) worth reporting for the "
        "whole-rollout LRS match; shorter true-longest repeats are reported as no match "
        "(default: 10).",
    )
    parser.add_argument("--degeneration-threshold", type=float, default=DEFAULT_DEGENERATION_THRESHOLD)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Rollouts to label in parallel via ProcessPoolExecutor (default: 1, serial). "
        "Labeling is embarrassingly parallel per-rollout.",
    )
    parser.add_argument(
        "--skip-normalized-lrs", action="store_true",
        help="Skip the comparison-only digit-normalized LRS variant (lrs_*_normalized "
        "columns) and its tokenizer load -- those columns are still written, filled "
        "with 'no match' placeholders.",
    )
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)
    domains = args.domains or sorted(paths.configured_domain_names(config))

    prompts_df = pd.read_parquet(paths.prompts_path(config))
    prompt_id_to_domain = dict(zip(prompts_df["prompt_id"], prompts_df["domain"]))

    digit_ids: Optional[Set[int]] = None
    if not args.skip_normalized_lrs:
        tokenizer_name = config.tokenizer_name or config.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        digit_ids = digit_token_ids(tokenizer)
        print(f"[normalized-lrs] {len(digit_ids)} digit token ids from {tokenizer_name!r}")

    all_labels_frames = []
    overall_start = time.time()
    for domain in domains:
        shard_in_path = paths.generations_shard_path(config, domain, 0)
        generations_df = pd.read_parquet(shard_in_path)

        start = time.time()
        labels_df = label_shard(
            generations_df,
            window_size=args.window_size,
            ttr_ngram=args.ttr_ngram,
            lrs_min_length=args.lrs_min_length,
            n_workers=args.workers,
            digit_token_ids=digit_ids,
        )
        elapsed = time.time() - start
        print(f"[{domain}] labeled {len(labels_df)} rollouts in {elapsed:.1f}s")

        shard_out_path = paths.labels_shard_path(config, domain, 0)
        write_shard_atomic(labels_df, shard_out_path)
        print(f"[{domain}] wrote {shard_out_path}")

        all_labels_frames.append(labels_df)

    total_elapsed = time.time() - overall_start
    total_rows = sum(len(f) for f in all_labels_frames)
    print(f"Labeled {total_rows} rollouts across {len(domains)} domains in {total_elapsed:.1f}s")

    all_labels_df = pd.concat(all_labels_frames, ignore_index=True)
    prompt_stats_df = aggregate_prompt_stats(
        all_labels_df, prompt_id_to_domain, degeneration_threshold=args.degeneration_threshold
    )
    prompt_stats_out_path = paths.prompt_stats_path(config)
    write_shard_atomic(prompt_stats_df, prompt_stats_out_path)
    print(f"Wrote {len(prompt_stats_df)} prompt_stats rows to {prompt_stats_out_path}")


if __name__ == "__main__":
    main()
