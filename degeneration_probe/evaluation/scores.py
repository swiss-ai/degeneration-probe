"""The interface between a scorer and the evaluation protocol.

Evaluation never receives a model. It receives per-token scores: one row per
rollout, carrying the rollout's identity and the probability the scorer
assigned to each of its tokens. Everything else follows from that.

The protocol is then blind to how a score was produced, so it applies
unchanged to any training recipe and to non-learned baselines such as a
repetition heuristic mapped into [0, 1], which puts probes and baselines in
the same table by construction. Producing scores costs a pass over the data on
a GPU; computing metrics from them is a small job on a CPU, so the protocol can
be extended or re-run whenever a question changes without recomputing a single
score. And because scores are stored rather than recomputed, two metrics
reported for the same run are guaranteed to describe the same numbers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

SCORE_COLUMNS = [
    "prompt_id",
    "rollout_idx",
    "domain",
    "split",
    "stop_reason",
    "num_tokens",
    "onset_position",
    "is_positive",
    "scores",
]


def validate_scores(frame: pd.DataFrame) -> None:
    """Refuse anything the protocol would otherwise misread silently."""
    missing = [column for column in SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Score table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Score table is empty")

    duplicated = frame.duplicated(["prompt_id", "rollout_idx"], keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, ["prompt_id", "rollout_idx"]].head().to_dict("records")
        raise ValueError(f"Duplicate rollout keys in the score table: {sample}")

    for row in frame.itertuples(index=False):
        scores = np.asarray(row.scores, dtype=np.float64)
        key = f"{row.prompt_id}/{row.rollout_idx}"
        if scores.size != int(row.num_tokens):
            raise ValueError(
                f"{key}: {scores.size} scores for a {int(row.num_tokens)}-token rollout"
            )
        if not np.isfinite(scores).all():
            raise ValueError(f"{key}: scores contain non-finite values")
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            raise ValueError(
                f"{key}: scores must lie in [0, 1], found [{scores.min()}, {scores.max()}]"
            )
        if row.is_positive:
            onset = row.onset_position
            if onset is None or pd.isna(onset):
                raise ValueError(f"{key}: a positive rollout needs an onset position")
            if not 0 <= int(onset) < int(row.num_tokens):
                raise ValueError(
                    f"{key}: onset {int(onset)} lies outside a {int(row.num_tokens)}-token rollout"
                )


def build_scores(records: Iterable[dict]) -> pd.DataFrame:
    """Assemble a score table from per-rollout dictionaries, then check it."""
    frame = pd.DataFrame(list(records))
    for column in ("prompt_id", "domain", "split", "stop_reason"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    frame["scores"] = [
        np.asarray(values, dtype=np.float16) for values in frame["scores"]
    ]
    frame = frame[SCORE_COLUMNS]
    validate_scores(frame)
    return frame


def write_scores(frame: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write atomically, so an interrupted scoring pass leaves no half table."""
    validate_scores(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_scores_", suffix=".parquet")
    os.close(handle)
    try:
        frame.to_parquet(staged, index=False)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return path


def read_scores(
    path: Union[str, Path], *, split: Optional[str] = None, validate: bool = True
) -> pd.DataFrame:
    """Load a score table, optionally restricted to one split."""
    frame = pd.read_parquet(path)
    if split is not None:
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No scored rollouts for split {split!r} in {path}")
    frame["scores"] = [np.asarray(values, dtype=np.float32) for values in frame["scores"]]
    if validate:
        validate_scores(frame)
    return frame
