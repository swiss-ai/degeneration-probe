"""Score the structural heuristics through the probe's own protocol.

The protocol takes per-token scores and is blind to how they were produced, so
the heuristics the labelling pipeline already computes can be scored as
detectors in their own right. That puts probes and baselines in the same four
views by construction, rather than in two tables whose numbers are only
loosely comparable.

Three baselines, each mapped into the unit interval by a monotone transform.
Monotone is all that is required: rank metrics are invariant to it, and every
threshold is chosen on validation data anyway, so the transform sets no
operating point of its own.

    repetition   a windowed repetition score, rising as text repeats
    entropy      the model's own predictive entropy, inverted, since
                 degeneration is confident rather than uncertain
    lrs          the longest-repeated-substring match, as a step from the
                 position where the repeat begins
    rep_l        the standard per-token repetition rate of the generation
                 literature: whether a token already appeared in the l before
                 it, smoothed over a trailing window

None of them needs a GPU, so this runs anywhere.

    python scripts/score_baselines.py --build-root <dataset> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import build_scores, write_scores

BASELINES = ("repetition", "entropy", "lrs", "rep_l")
# Welleck et al. report rep/l at l = 128, and the trailing average is taken over
# the same width so the score has one scale rather than two.
REP_L_LOOKBACK = 128
REP_L_WINDOW = 128


def repetition_scores(values, num_tokens: int) -> np.ndarray:
    """A repetition score, already bounded, aligned and gap-filled."""
    scores = np.full(num_tokens, np.nan, dtype=np.float64)
    supplied = np.asarray(list(values)[:num_tokens], dtype=np.float64)
    scores[: supplied.size] = supplied
    return _fill_gaps(np.clip(scores, 0.0, 1.0))


def entropy_scores(values, num_tokens: int) -> np.ndarray:
    """Inverted predictive entropy: a loop is confident, not uncertain.

    ``1 / (1 + H)`` is monotone decreasing in entropy and lands in (0, 1]
    without needing a population to normalize against, so a score does not
    change meaning when the split it was computed on does.
    """
    entropy = np.full(num_tokens, np.nan, dtype=np.float64)
    supplied = np.asarray(list(values)[:num_tokens], dtype=np.float64)
    entropy[: supplied.size] = supplied
    return _fill_gaps(1.0 / (1.0 + np.clip(entropy, 0.0, None)))


def lrs_scores(first_start: Optional[float], length: Optional[float], num_tokens: int) -> np.ndarray:
    """A step at the position where the longest repeated substring begins."""
    scores = np.zeros(num_tokens, dtype=np.float64)
    if length is None or pd.isna(length) or int(length) == 0:
        return scores
    if first_start is None or pd.isna(first_start):
        return scores
    start = int(np.clip(int(first_start), 0, num_tokens))
    scores[start:] = 1.0
    return scores


def rep_l_scores(
    token_ids,
    num_tokens: int,
    lookback: int = REP_L_LOOKBACK,
    window: int = REP_L_WINDOW,
) -> np.ndarray:
    """The repetition rate this literature already uses at token level.

    Welleck et al. define ``rep/l`` as the share of next-token predictions that
    already occur in the previous ``l`` tokens. It is backward looking by
    construction, which is what makes it usable as an online baseline: the
    windowed repetition score of the labelling pipeline reads the window *ahead*
    of the token it labels, which is the right choice for annotating a finished
    rollout and an invalid one for a scorer a live monitor could run.

    Two departures from the original are worth stating. The indicator is taken
    over the token that was emitted rather than over the model's top-1
    prediction, because a monitor observes the emitted token and the top-1 is
    not stored. And the per-token indicator is averaged over a trailing window
    rather than reported as one number per generation, because the protocol
    needs a position; a bare indicator is binary and would tie almost every
    rollout at the top of its range, which is exactly what stops the
    longest-repeated-substring baseline from spending a false-alarm budget.
    """
    ids = np.asarray(list(token_ids)[:num_tokens], dtype=np.int64)
    length = ids.size
    if length == 0:
        return np.zeros(num_tokens, dtype=np.float64)

    # One pass with a sliding multiset: at token t the window holds the ids at
    # positions [t - lookback, t), so the whole signal costs O(n) rather than
    # rescanning the lookback at every position.
    repeated = np.zeros(length, dtype=np.float64)
    counts: Dict[int, int] = {}
    for position in range(length):
        if position:
            entering = int(ids[position - 1])
            counts[entering] = counts.get(entering, 0) + 1
        leaving_index = position - 1 - lookback
        if leaving_index >= 0:
            leaving = int(ids[leaving_index])
            remaining = counts[leaving] - 1
            if remaining:
                counts[leaving] = remaining
            else:
                del counts[leaving]
        repeated[position] = 1.0 if counts.get(int(ids[position]), 0) else 0.0

    cumulative = np.concatenate([[0.0], np.cumsum(repeated)])
    ends = np.arange(length) + 1
    starts = np.maximum(0, ends - window)
    scores = (cumulative[ends] - cumulative[starts]) / (ends - starts)

    padded = np.zeros(num_tokens, dtype=np.float64)
    padded[:length] = scores
    return padded


def _fill_gaps(scores: np.ndarray) -> np.ndarray:
    """Carry the last defined value forward; lead with the first one."""
    if np.isnan(scores).all():
        return np.zeros_like(scores)
    indices = np.where(~np.isnan(scores), np.arange(scores.size), 0)
    np.maximum.accumulate(indices, out=indices)
    filled = scores[indices]
    first = np.flatnonzero(~np.isnan(scores))[0]
    filled[:first] = scores[first]
    return filled


def build_baseline_scores(
    build_root: Path, baseline: str, split: str
) -> pd.DataFrame:
    onset = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    onset = onset[onset["split"] == split]
    if onset.empty:
        raise ValueError(f"No rollouts in split {split!r}")

    # Three baselines are precomputed columns of the label store. rep/l is not:
    # it is a function of the generated ids themselves, which live beside the
    # text rather than beside the labels.
    source = "generations" if baseline == "rep_l" else "labels"
    columns = ["prompt_id", "rollout_idx"]
    if baseline == "rep_l":
        columns += ["generated_token_ids"]
    elif baseline == "repetition":
        columns += ["repetition_score"]
    elif baseline == "entropy":
        columns += ["entropy"]
    else:
        columns += [
            "lrs_first_start_normalized_growing",
            "lrs_length_normalized_growing",
        ]

    records = []
    for domain in sorted(onset["domain"].astype(str).unique()):
        labels = pd.concat(
            [
                pd.read_parquet(path, columns=columns)
                for path in sorted((build_root / source / domain).glob("*.parquet"))
            ],
            ignore_index=True,
        )
        merged = onset[onset["domain"].astype(str) == domain].merge(
            labels, on=["prompt_id", "rollout_idx"], how="inner", validate="one_to_one"
        )
        for row in merged.itertuples(index=False):
            tokens = int(row.num_tokens)
            if baseline == "rep_l":
                scores = rep_l_scores(row.generated_token_ids, tokens)
            elif baseline == "repetition":
                scores = repetition_scores(row.repetition_score, tokens)
            elif baseline == "entropy":
                scores = entropy_scores(row.entropy, tokens)
            else:
                scores = lrs_scores(
                    row.lrs_first_start_normalized_growing,
                    row.lrs_length_normalized_growing,
                    tokens,
                )
            positive = bool(row.is_positive) and pd.notna(row.onset_position)
            records.append(
                {
                    "prompt_id": row.prompt_id,
                    "rollout_idx": int(row.rollout_idx),
                    "domain": str(row.domain),
                    "split": str(row.split),
                    "stop_reason": str(row.stop_reason),
                    "num_tokens": tokens,
                    "onset_position": float(row.onset_position)
                    if pd.notna(row.onset_position)
                    else None,
                    "is_positive": positive,
                    "scores": scores.astype(np.float16),
                }
            )
    return build_scores(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES))
    parser.add_argument(
        "--splits", nargs="*", default=["val", "test_indomain", "test_heldout_domains"]
    )
    args = parser.parse_args()

    for baseline in args.baselines:
        if baseline not in BASELINES:
            raise SystemExit(f"Unknown baseline {baseline!r}; expected one of {list(BASELINES)}")
        for split in args.splits:
            frame = build_baseline_scores(args.build_root, baseline, split)
            path = write_scores(frame, args.out_dir / baseline / "scores" / f"{split}.parquet")
            positives = int(frame["is_positive"].sum())
            print(
                f"  {baseline:<11} {split:<22} {len(frame):>6} rollouts "
                f"({positives} positive), {int(frame['num_tokens'].sum()):>10,} tokens -> {path}"
            )


if __name__ == "__main__":
    main()
