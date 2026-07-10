"""Phase 2 of the dataset-v2 pilot: per-token degeneration labels.

For every rollout in ``paths.generations_shard_path(config, domain, 0)`` (written
by ``generate.py``) this module computes three per-token degeneration signals,
aligned to the same token positions, so the team can compare which one best
tracks actual repetition:

1. ``repetition_score`` -- sliding-window ``1 - TTR`` over bigrams (n=2).
   Ported from the reference implementation at
   ``origin/main:src/degeneration_probe/probe/data_loader.py``
   (``_ngram_ttr`` / ``sliding_window_repetition``): for each position ``t``,
   ``1 - (distinct 2-grams in tokens[t:t+window_size]) / (window_size - 1)``.
   Positions where a full window doesn't fit at the end of the sequence are
   NaN (masked), matching that reference's convention exactly.

2. ``lrs_window_score`` -- Longest Repeated Substring, made per-token by
   applying the (naive, per the given spec) whole-sequence ``lrs_score``
   formula to each length-``window_size`` window instead of the whole
   sequence, using the *same* NaN-at-the-tail convention as (1) so the two
   signals are directly comparable position-for-position.

3. ``entropy`` -- pass-through of Phase 1's ``per_token_entropy`` column
   (already per-token, already aligned to ``generated_token_ids``).

Output: ``labels/<domain>/shard_00000.parquet`` (via
``paths.labels_shard_path``) with columns
``[prompt_id, rollout_idx, repetition_score, lrs_window_score, entropy]``,
one row per rollout.

This module also aggregates across ALL domains into
``prompt_stats/prompt_stats.parquet`` (via ``paths.prompt_stats_path``): one
row per prompt_id summarizing its ``n_rollouts_per_prompt`` (10) rollouts --
see ``aggregate_prompt_stats`` for exactly how "mean"/"max"/"degeneration_rate"
are defined across the two levels (per-rollout, then per-prompt).

Performance note on LRS
------------------------
The per-window LRS computation is the naive O(window_size^2)-ish algorithm
from the given spec, run once per token position (i.e. up to
``num_tokens - window_size + 1`` times per rollout) rather than once per
rollout. At ``window_size=256`` (up from the pilot's 32) this is roughly
64x more work per window than the pilot used; see the module's CLI output
for the measured wall-clock time on the real data, and benchmark on a small
``--domains`` subset before launching a full run if that matters.
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
from degeneration_probe.dataset_gen.config import PilotDatasetConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset_gen" / "pilot_v1.yaml"

DEFAULT_WINDOW_SIZE = 256
DEFAULT_TTR_NGRAM = 2
DEFAULT_LRS_MIN_LENGTH = 10
# Matches configs/dataset/repetition_instruct_token_level.yaml's degeneration_threshold.
DEFAULT_DEGENERATION_THRESHOLD = 0.8

LABEL_COLUMNS = ["prompt_id", "rollout_idx", "repetition_score", "lrs_window_score", "entropy"]

PROMPT_STATS_COLUMNS = [
    "prompt_id",
    "domain",
    "n_rollouts",
    "mean_repetition_score",
    "max_repetition_score",
    "mean_lrs_window_score",
    "max_lrs_window_score",
    "degeneration_rate",
]


# --- TTR window score (ported from origin/main:src/degeneration_probe/probe/data_loader.py) --

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


# --- LRS (longest repeated substring) window score ---------------------------

def lrs_score(tokens: Sequence[int], min_length: int = 10) -> float:
    """Naive O(n^2)-ish longest-repeated-non-overlapping-substring fraction.

    Exactly the reference implementation given in the task spec: for every
    pair of start positions (i, j) with j >= i + min_length, extend a match
    while tokens line up, tokens don't overlap (i + length < j), and both
    indices stay in bounds. Returns the longest such match length / n.
    """
    n = len(tokens)
    max_len = 0
    for i in range(n):
        for j in range(i + min_length, n):
            length = 0
            while (
                j + length < n
                and i + length < j
                and tokens[i + length] == tokens[j + length]
            ):
                length += 1
            max_len = max(max_len, length)
    return max_len / n


def sliding_window_lrs(
    token_ids: Sequence[int],
    window_size: int,
    min_length: int = DEFAULT_LRS_MIN_LENGTH,
) -> List[float]:
    """Per-position LRS score over token_ids[t : t + window_size], NaN-masked.

    Same tail convention as `sliding_window_repetition`: positions where a
    full window doesn't fit at the end of the sequence are NaN.
    """
    out: List[float] = []
    total = len(token_ids)
    for t in range(total):
        end = t + window_size
        if end > total:
            out.append(float("nan"))
        else:
            out.append(lrs_score(token_ids[t:end], min_length=min_length))
    return out


# --- per-rollout / per-shard labeling ------------------------------------------

def label_rollout(
    generated_token_ids: Sequence[int],
    per_token_entropy: Sequence[float],
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttr_ngram: int = DEFAULT_TTR_NGRAM,
    lrs_min_length: int = DEFAULT_LRS_MIN_LENGTH,
) -> Dict[str, List[float]]:
    """Compute the three aligned per-token signals for one rollout."""
    token_ids = [int(t) for t in generated_token_ids]
    repetition = sliding_window_repetition(token_ids, window_size, n=ttr_ngram)
    lrs = sliding_window_lrs(token_ids, window_size, min_length=lrs_min_length)
    entropy = [float(e) for e in per_token_entropy]
    return {"repetition_score": repetition, "lrs_window_score": lrs, "entropy": entropy}


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
    return {
        "prompt_id": prompt_id,
        "rollout_idx": rollout_idx,
        "repetition_score": labels["repetition_score"],
        "lrs_window_score": labels["lrs_window_score"],
        "entropy": labels["entropy"],
    }


def label_shard(
    generations_df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttr_ngram: int = DEFAULT_TTR_NGRAM,
    lrs_min_length: int = DEFAULT_LRS_MIN_LENGTH,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Label every rollout in one domain's generations shard.

    The naive LRS computation (see module docstring) makes each rollout's
    labeling independent and CPU-heavy, so with ``n_workers > 1`` rollouts
    are farmed out to a ``ProcessPoolExecutor``. ``executor.map`` returns
    results in the same order as the input tasks (not completion order), so
    the returned DataFrame's row order always matches ``generations_df``'s.
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

def _rollout_summary(repetition: Sequence[float], lrs: Sequence[float]) -> Dict[str, float]:
    """Reduce one rollout's per-token arrays to a (mean, max) pair for each signal.

    NaN-tail positions are ignored (np.nanmean/np.nanmax). If a rollout has no
    valid windows at all (shorter than window_size -- doesn't happen in the
    real pilot data, but handled defensively), both reduce to NaN rather than
    raising.
    """
    repetition_arr = np.asarray(repetition, dtype=float)
    lrs_arr = np.asarray(lrs, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice
        mean_repetition = (
            float(np.nanmean(repetition_arr)) if np.any(~np.isnan(repetition_arr)) else float("nan")
        )
        max_repetition = (
            float(np.nanmax(repetition_arr)) if np.any(~np.isnan(repetition_arr)) else float("nan")
        )
        mean_lrs = float(np.nanmean(lrs_arr)) if np.any(~np.isnan(lrs_arr)) else float("nan")
        max_lrs = float(np.nanmax(lrs_arr)) if np.any(~np.isnan(lrs_arr)) else float("nan")

    return {
        "mean_repetition_score": mean_repetition,
        "max_repetition_score": max_repetition,
        "mean_lrs_window_score": mean_lrs,
        "max_lrs_window_score": max_lrs,
    }


def aggregate_prompt_stats(
    labels_df: pd.DataFrame,
    prompt_id_to_domain: Dict[str, str],
    degeneration_threshold: float = DEFAULT_DEGENERATION_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate a (multi-domain) labels table to one row per prompt_id.

    For each rollout, first reduce its per-token repetition/lrs arrays to a
    (mean, max) pair (NaN-tail positions ignored). Then, per prompt_id, across
    its rollouts:
      - mean_repetition_score / mean_lrs_window_score: mean of the per-rollout
        means (NaN rollouts, i.e. ones with no valid window, are ignored).
      - max_repetition_score / max_lrs_window_score: max of the per-rollout
        maxes.
      - degeneration_rate: fraction of rollouts whose max_repetition_score
        exceeds `degeneration_threshold` (a rollout with an undefined,
        NaN max never counts as exceeding it).
    """
    per_rollout_records = []
    for row in labels_df.itertuples(index=False):
        summary = _rollout_summary(row.repetition_score, row.lrs_window_score)
        summary["prompt_id"] = row.prompt_id
        summary["rollout_idx"] = int(row.rollout_idx)
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
            mean_lrs = float(group["mean_lrs_window_score"].mean(skipna=True))
            max_lrs = float(group["max_lrs_window_score"].max(skipna=True))

        prompt_records.append(
            {
                "prompt_id": prompt_id,
                "domain": prompt_id_to_domain.get(prompt_id),
                "n_rollouts": n_rollouts,
                "mean_repetition_score": mean_repetition,
                "max_repetition_score": max_repetition,
                "mean_lrs_window_score": mean_lrs,
                "max_lrs_window_score": max_lrs,
                "degeneration_rate": n_degenerating / n_rollouts,
            }
        )

    return pd.DataFrame.from_records(prompt_records, columns=PROMPT_STATS_COLUMNS)


# --- CLI entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to a PilotDatasetConfig YAML file (default: configs/dataset_gen/pilot_v1.yaml).",
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="Subset of domains to label (default: all configured domains).",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--ttr-ngram", type=int, default=DEFAULT_TTR_NGRAM)
    parser.add_argument("--lrs-min-length", type=int, default=DEFAULT_LRS_MIN_LENGTH)
    parser.add_argument("--degeneration-threshold", type=float, default=DEFAULT_DEGENERATION_THRESHOLD)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Rollouts to label in parallel via ProcessPoolExecutor (default: 1, serial). "
        "The naive LRS computation is CPU-heavy and embarrassingly parallel per-rollout.",
    )
    args = parser.parse_args()

    config = PilotDatasetConfig.from_yaml(args.config)
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
