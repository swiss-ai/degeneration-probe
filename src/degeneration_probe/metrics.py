"""Utilities for chunk-based repetition and TTR analysis."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def chunk_token_ids(token_ids: Sequence[int], chunk_size: int) -> List[List[int]]:
    """Split token ids into non-overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    return [list(token_ids[i : i + chunk_size]) for i in range(0, len(token_ids), chunk_size)]


def compute_ngram_ttr(token_ids: Sequence[int], n: int) -> float:
    """Compute type-token ratio over n-grams for a token-id sequence."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(token_ids) < n:
        return 1.0

    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def compute_ngram_repetition(token_ids: Sequence[int], n: int) -> float:
    """Compute repetition score as 1 - TTR over n-grams."""
    return 1.0 - compute_ngram_ttr(token_ids, n)


def summarize_chunks(
    token_ids: Sequence[int],
    n_values: Iterable[int],
    chunk_size: int,
) -> dict:
    """Return per-chunk and aggregated metrics for several n-gram sizes."""
    chunks = chunk_token_ids(token_ids, chunk_size)
    metrics: dict[str, dict] = {}

    for n in n_values:
        per_chunk = []
        for idx, chunk in enumerate(chunks):
            ttr = compute_ngram_ttr(chunk, n)
            per_chunk.append(
                {
                    "chunk_index": idx,
                    "chunk_length": len(chunk),
                    "ttr": ttr,
                    "repetition": 1.0 - ttr,
                }
            )

        ttr_values = [item["ttr"] for item in per_chunk]
        repetition_values = [item["repetition"] for item in per_chunk]
        metrics[str(n)] = {
            "per_chunk": per_chunk,
            "min_ttr": min(ttr_values) if ttr_values else 1.0,
            "max_repetition": max(repetition_values) if repetition_values else 0.0,
            "mean_ttr": sum(ttr_values) / len(ttr_values) if ttr_values else 1.0,
            "mean_repetition": sum(repetition_values) / len(repetition_values) if repetition_values else 0.0,
        }

    return {
        "num_tokens": len(token_ids),
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "metrics_by_n": metrics,
    }
