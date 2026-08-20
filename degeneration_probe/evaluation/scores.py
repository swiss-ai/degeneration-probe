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

# A capped answer whose onset could not be resolved is not a healthy answer. The
# judge either could not be run on it or its quote could not be located, so
# nothing is known about it, and counting it among the negatives spends part of a
# false-alarm budget on an answer that is probably a positive. Training already
# excludes these, because a capped answer with no frontier has no defined target.
# `not_degenerating` is deliberately absent: the judge did run on those and ruled
# them healthy, so they belong among the negatives.
UNJUDGED_RESOLUTIONS = ("judge_failed", "not_found")

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
    # float32, not float16. Half precision has a spacing of about 5e-4 just
    # below one, so every score within that distance of the top of the range
    # collapses onto it. Ties at a scorer's ceiling are what make a small
    # false-alarm budget unspendable, so storing at half precision manufactures
    # the very failure the protocol reports: the windowed entropy baseline moves
    # from 0.2% of healthy answers tied at its ceiling to 1.2%, which is the
    # difference between a 1% budget that can be spent and one that cannot.
    frame["scores"] = [
        np.asarray(values, dtype=np.float32) for values in frame["scores"]
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


def unjudged_rollouts(onset_labels_path: Union[str, Path]) -> set:
    """The rollouts no class can be assigned to, as (prompt_id, rollout_idx).

    Read from the onset label table rather than inferred from a score table,
    because the distinction that matters is *why* a capped answer has no frontier
    and only the label table records that.
    """
    labels = pd.read_parquet(
        onset_labels_path, columns=["prompt_id", "rollout_idx", "onset_resolution"]
    )
    unjudged = labels[labels["onset_resolution"].isin(UNJUDGED_RESOLUTIONS)]
    return {
        (str(row.prompt_id), int(row.rollout_idx))
        for row in unjudged.itertuples(index=False)
    }


def drop_unjudged(frame: pd.DataFrame, unjudged: set) -> tuple:
    """Remove the rollouts nothing is known about, and say how many went.

    Returns the frame and the count dropped, so a caller can report it rather
    than change a population silently.
    """
    if not unjudged:
        return frame, 0
    keys = [
        (str(prompt), int(index))
        for prompt, index in zip(frame["prompt_id"], frame["rollout_idx"])
    ]
    keep = np.array([key not in unjudged for key in keys])
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def read_scores(
    path: Union[str, Path],
    *,
    split: Optional[str] = None,
    validate: bool = True,
    unjudged: Optional[set] = None,
) -> pd.DataFrame:
    """Load a score table, optionally restricted to one split.

    ``unjudged`` names rollouts to leave out of the population entirely. Pass the
    result of :func:`unjudged_rollouts`; ``None`` keeps every scored rollout, which
    counts a capped answer the judge could not rule on as healthy.
    """
    frame = pd.read_parquet(path)
    if split is not None:
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No scored rollouts for split {split!r} in {path}")
    if unjudged:
        frame, _ = drop_unjudged(frame, unjudged)
        if frame.empty:
            raise ValueError(f"Every scored rollout in {path} is unjudged")
    frame["scores"] = [np.asarray(values, dtype=np.float32) for values in frame["scores"]]
    if validate:
        validate_scores(frame)
    return frame
