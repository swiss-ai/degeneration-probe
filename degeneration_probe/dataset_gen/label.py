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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"

DEFAULT_WINDOW_SIZE = 256
DEFAULT_TTR_NGRAM = 2
DEFAULT_LRS_MIN_LENGTH = 10
# Matches configs/dataset/repetition_instruct_token_level.yaml's degeneration_threshold.
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


def _no_lrs_match() -> Dict[str, object]:
    return {
        "lrs_length": 0,
        "lrs_score": 0.0,
        "lrs_first_start": None,
        "lrs_second_start": None,
        "lrs_gap": None,
        "lrs_repeated_token_ids": [],
        "lrs_period": None,
        "lrs_period_repeat_count": None,
        "lrs_region_starts": [],
        "lrs_region_ends": [],
        "lrs_unit_token_ids": [],
    }


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


def _extend_periodic_region(tokens: Sequence[int], start: int, period: int) -> Tuple[int, int]:
    """Given that `tokens[start : start + period]` repeating with period `period`
    describes at least part of the sequence, extend `[start, end)` as far as
    that exact periodicity actually holds across the whole sequence in both
    directions.

    Only walks *adjacent* period-steps, so it only grows across genuinely
    back-to-back repetition -- it stops the instant unrelated content
    intervenes. That's intentional: see `_periodic_regions_for_match` for why
    a single call to this from only one of the two occurrences isn't enough
    on its own.
    """
    n = len(tokens)
    end = start + period
    while end < n and tokens[end] == tokens[end - period]:
        end += 1
    while start > 0 and tokens[start - 1] == tokens[start - 1 + period]:
        start -= 1
    return start, end


def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/touching (start, end) spans into a sorted, disjoint list."""
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _periodic_regions_for_match(
    tokens: Sequence[int], first_start: int, second_start: int, period: int
) -> List[Tuple[int, int]]:
    """The full set of disjoint spans over which the matched `period`-token unit
    repeats, covering *both* known occurrences.

    `_extend_periodic_region` only grows through back-to-back repetition, so
    seeding it from `first_start` alone silently loses the second occurrence
    whenever there's a real gap of unrelated content between the two matches
    (e.g. `period == lrs_length`, i.e. the matched chunk has no smaller
    internal repetition, and the two occurrences are genuinely far apart --
    the gap is then not itself a multiple of `period`, so extending from
    `first_start` stops dead at the end of the first occurrence and never
    reaches the second). Seeding from *both* occurrences and merging fixes
    this: when the run really is one contiguous periodic stretch (gap is a
    multiple of `period`), both seeds extend into and merge onto the same
    span; when it's two genuinely separate occurrences, both seeds are kept
    as distinct spans instead of one silently vanishing.
    """
    span_a = _extend_periodic_region(tokens, first_start, period)
    span_b = _extend_periodic_region(tokens, second_start, period)
    return _merge_spans([span_a, span_b])


# An atomic chunk's *own* recurrence interval (the gap between the two
# occurrences the binary search happened to find) has nothing to do with its
# internal period -- there's no adjacent-step relationship to walk via
# `_extend_periodic_region`. But that same interval is a good guess for where
# *further* copies live (e.g. a model looping through the same paragraph with
# an incrementing step counter spliced in every cycle), so
# `_gap_aligned_occurrences` checks every multiple of that gap directly.
_GAP_MATCH_MIN_FRACTION = 0.8


def _gap_aligned_occurrences(
    tokens: Sequence[int],
    first_start: int,
    gap: int,
    span_length: int,
    min_match_fraction: float = _GAP_MATCH_MIN_FRACTION,
) -> Tuple[List[Tuple[int, int]], int]:
    """Look for further copies of `tokens[first_start : first_start + span_length]`
    at every other multiple of `gap` from `first_start` (both directions), not
    just at `first_start` and `first_start + gap` themselves.

    A copy rarely matches byte-for-byte: if the model is looping with some
    slowly-changing element spliced into each cycle (an incrementing step
    counter, say), only *most* of the span recurs exactly. So each candidate
    position is scored by its longest common prefix with the anchor span
    rather than requiring full equality, and a candidate only counts if that
    prefix covers at least `min_match_fraction` of `span_length` -- enough to
    be confident it's really the same content, not a coincidence.

    All accepted occurrences (including the original anchor) are then
    uniformly trimmed to the *shortest* common-prefix length found among
    them, so every reported span reflects only the part that's genuinely
    stable across all of them -- e.g. if a trailing token happens to match in
    some cycles by luck (two step numbers sharing a leading digit) but not
    others, it's dropped everywhere rather than kept in some spans and not
    others.

    Returns `(regions, core_length)`; `core_length <= span_length`, and
    `regions` always contains at least the trimmed anchor itself.
    """
    n = len(tokens)
    anchor = tokens[first_start : first_start + span_length]
    threshold = max(1, round(min_match_fraction * span_length))

    k_min = -(first_start // gap)
    k_max = (n - 1 - first_start) // gap

    candidates: List[Tuple[int, int]] = []
    for k in range(k_min, k_max + 1):
        c = first_start + k * gap
        limit = min(span_length, n - c)
        match_len = 0
        while match_len < limit and tokens[c + match_len] == anchor[match_len]:
            match_len += 1
        if match_len >= threshold:
            candidates.append((c, match_len))

    core_length = min(match_len for _c, match_len in candidates)
    regions = [(c, c + core_length) for c, _match_len in candidates]
    return _merge_spans(regions), core_length


def find_longest_repeated_substring(
    token_ids: Sequence[int],
    min_length: int = DEFAULT_LRS_MIN_LENGTH,
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
    `_minimal_period` / `_periodic_regions_for_match`: on fully-degenerate,
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

    For that atomic case, `lrs_gap` (the interval between the one pair the
    binary search happened to find) is also used as a probe for *further*
    copies elsewhere in the sequence -- see `_gap_aligned_occurrences`. This
    matters a lot for loops that repeat the same content but splice in some
    slowly-changing element each cycle (a step counter that keeps
    incrementing, say): the binary search can only ever match the stable
    part shared between exactly two cycles, so left alone it silently
    undercounts a 9-cycle loop as "repeated twice". When this finds more
    occurrences, `lrs_period`/`lrs_unit_token_ids` are trimmed to the longest
    prefix shared by *all* of them (not just the original pair), and
    `lrs_period_repeat_count` reflects the true occurrence count.

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
    """
    n = len(token_ids)
    tokens = list(token_ids)

    if n < 2 * min_length:
        return _no_lrs_match()

    # Prefix hashes / powers are independent of L -- computed once, then any
    # window's hash is O(1): hash(tokens[i:i+L]) = prefix[i+L] - prefix[i] * power[L].
    prefix = [0] * (n + 1)
    power = [1] * (n + 1)
    for idx in range(n):
        prefix[idx + 1] = (prefix[idx] * _HASH_BASE + tokens[idx] + 1) % _HASH_MOD
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
                if i - p >= length and tokens[p : p + length] == tokens[i : i + length]:
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
        return _no_lrs_match()

    first_start, second_start = best_pair
    matched_span = tokens[first_start : first_start + best_length]
    period = _minimal_period(matched_span)
    if period == best_length:
        # Atomic chunk (no internal sub-repetition): its own recurrence gap,
        # not its (nonexistent) internal period, is the right step size to
        # search for further copies -- see _gap_aligned_occurrences.
        gap = second_start - first_start
        regions, period = _gap_aligned_occurrences(tokens, first_start, gap, best_length)
    else:
        regions = _periodic_regions_for_match(tokens, first_start, second_start, period)
    region_starts = [s for s, _e in regions]
    region_ends = [e for _s, e in regions]
    first_region_start = region_starts[0]

    return {
        "lrs_length": best_length,
        "lrs_score": best_length / n,
        "lrs_first_start": first_start,
        "lrs_second_start": second_start,
        "lrs_gap": second_start - first_start,
        "lrs_repeated_token_ids": matched_span,
        "lrs_period": period,
        "lrs_period_repeat_count": sum((e - s) // period for s, e in regions),
        "lrs_region_starts": region_starts,
        "lrs_region_ends": region_ends,
        "lrs_unit_token_ids": tokens[first_region_start : first_region_start + period],
    }


# --- per-rollout / per-shard labeling ------------------------------------------

def label_rollout(
    generated_token_ids: Sequence[int],
    per_token_entropy: Sequence[float],
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttr_ngram: int = DEFAULT_TTR_NGRAM,
    lrs_min_length: int = DEFAULT_LRS_MIN_LENGTH,
) -> Dict[str, object]:
    """Compute the per-token repetition/entropy signals plus the whole-rollout LRS match."""
    token_ids = [int(t) for t in generated_token_ids]
    repetition = sliding_window_repetition(token_ids, window_size, n=ttr_ngram)
    entropy = [float(e) for e in per_token_entropy]
    lrs = find_longest_repeated_substring(token_ids, min_length=lrs_min_length)
    return {"repetition_score": repetition, "entropy": entropy, **lrs}


def _label_rollout_task(task: Tuple[str, int, Sequence[int], Sequence[float], int, int, int]) -> dict:
    """Picklable ProcessPoolExecutor worker: label one rollout, tagged with its keys."""
    prompt_id, rollout_idx, generated_token_ids, per_token_entropy, window_size, ttr_ngram, lrs_min_length = task
    labels = label_rollout(
        generated_token_ids,
        per_token_entropy,
        window_size=window_size,
        ttr_ngram=ttr_ngram,
        lrs_min_length=lrs_min_length,
    )
    return {"prompt_id": prompt_id, "rollout_idx": rollout_idx, **labels}


def label_shard(
    generations_df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttr_ngram: int = DEFAULT_TTR_NGRAM,
    lrs_min_length: int = DEFAULT_LRS_MIN_LENGTH,
    n_workers: int = 1,
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
        help="Path to a DatasetGenConfig YAML file (default: configs/dataset/degeneration-dataset-apertus-8b-instruct.yaml).",
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
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)
    domains = args.domains or sorted(paths.configured_domain_names(config))

    prompts_df = pd.read_parquet(paths.prompts_path(config))
    prompt_id_to_domain = dict(zip(prompts_df["prompt_id"], prompts_df["domain"]))

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
