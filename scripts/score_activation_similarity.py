"""Score how much the residual stream has started repeating itself.

A trained probe is only interesting if it beats noticing that the model's own
hidden states have begun revisiting where they have already been. That question
needs no training at all: at each token, take the cosine similarity between the
current hidden state and each of the previous ones inside a trailing window, and
keep the largest. A state that has returned to somewhere it has been scores high,
and one moving through new territory scores low.

This is the untrained core of the signal Yu et al. feed to a classifier. Their
detector is not this: it collects per-token maxima of activation similarity
across the decoder's MLP layers, sorts them, concatenates the sorted and unsorted
vectors, and passes the result to a three-layer network that scores a whole
response after a 400-token warm-up. What is shared is the primitive. Calling this
baseline theirs would overstate the correspondence, so it is named for what it
computes.

Read at one layer the cache holds, so the cost is one strided read per rollout
rather than a forward pass. Every variant is computed in the same pass, because
opening and reading the file dominates and the arithmetic afterwards is free.

    sbatch cluster/activation_similarity.sbatch --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.data.activation_store import load_probe_layers
from degeneration_probe.evaluation.scores import build_scores, write_scores

# The same 256 tokens the windowed repetition and entropy baselines look back
# over, so a difference between them is a difference in what is being measured
# rather than in how far back it is measured.
DEFAULT_WINDOW = 256
# How many of the most recent tokens to skip. Neighbouring residual streams are
# similar for reasons that have nothing to do with a loop, so a scorer that
# compares a token only to its immediate predecessor reports that similarity
# everywhere. Several are computed and the choice is made on whether the
# resulting scorer can spend a false-alarm budget.
DEFAULT_MIN_LAGS = (1, 16, 64)
DEFAULT_LAYERS = (4, 8, 12, 15, 20, 30)
ROW_BLOCK = 256


def trailing_max_similarity(
    hidden: torch.Tensor, window: int, min_lag: int, row_block: int = ROW_BLOCK
) -> np.ndarray:
    """Per token, the largest cosine similarity to a state in its trailing window.

    Token ``t`` is compared against tokens ``[t - window, t - min_lag]``. The
    opening tokens have no such neighbour and score zero rather than being
    dropped, so the returned array aligns with the rollout one to one.
    """
    if window < 1:
        raise ValueError(f"window must be at least one token, got {window}")
    if min_lag < 1:
        raise ValueError(f"min_lag must be at least one token, got {min_lag}")
    tokens = int(hidden.shape[0])
    # The first ``min_lag`` tokens have nothing behind them to be similar to.
    # They start at the bottom of the cosine range rather than at zero, so that
    # after the map onto the unit interval they sit at the bottom of the score
    # range and never trip a threshold. Scoring them in the middle would make
    # every rollout alarm on its own opening tokens.
    similarity = torch.full((tokens,), -1.0, dtype=torch.float32, device=hidden.device)
    if tokens <= min_lag:
        return ((similarity + 1.0) / 2.0).cpu().numpy()

    unit = torch.nn.functional.normalize(hidden.to(torch.float32), dim=1)
    for start in range(0, tokens, row_block):
        stop = min(start + row_block, tokens)
        first_column = max(0, start - window)
        last_column = stop - min_lag  # exclusive
        if last_column <= first_column:
            continue
        block = unit[start:stop] @ unit[first_column:last_column].T
        rows = torch.arange(start, stop, device=hidden.device).unsqueeze(1)
        columns = torch.arange(first_column, last_column, device=hidden.device).unsqueeze(0)
        allowed = (columns <= rows - min_lag) & (columns >= rows - window)
        block = block.masked_fill(~allowed, float("-inf"))
        best = block.max(dim=1).values
        similarity[start:stop] = torch.where(
            torch.isfinite(best), best, torch.full_like(best, -1.0)
        )
    # Cosine similarity lands in [-1, 1]; the protocol reads scores on the unit
    # interval. The map is monotone, so it sets no operating point of its own.
    return ((similarity + 1.0) / 2.0).clamp_(0.0, 1.0).cpu().numpy()


def score_split(
    build_root: Path,
    activations_root: Path,
    split: str,
    layers: Sequence[int],
    window: int,
    min_lags: Sequence[int],
    device: str,
    limit: int | None = None,
) -> dict:
    onset = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    onset = onset[onset["split"] == split].reset_index(drop=True)
    if onset.empty:
        raise ValueError(f"No rollouts in split {split!r}")
    if limit:
        onset = onset.head(limit)

    records: dict = {(layer, lag): [] for layer in layers for lag in min_lags}
    for position, row in enumerate(onset.itertuples(index=False)):
        stack = load_probe_layers(
            activations_root,
            str(row.domain),
            str(row.prompt_id),
            int(row.rollout_idx),
            probe_layers=list(layers),
            expected_tokens=None,
        ).to(device)
        tokens = int(row.num_tokens)
        positive = bool(row.is_positive) and pd.notna(row.onset_position)
        identity = {
            "prompt_id": str(row.prompt_id),
            "rollout_idx": int(row.rollout_idx),
            "domain": str(row.domain),
            "split": str(row.split),
            "stop_reason": str(row.stop_reason),
            "num_tokens": tokens,
            "onset_position": float(row.onset_position) if pd.notna(row.onset_position) else None,
            "is_positive": positive,
        }
        for index, layer in enumerate(layers):
            hidden = stack[index]
            for lag in min_lags:
                scores = trailing_max_similarity(hidden, window, lag)
                # The cache is written from the completion, so a length
                # disagreement is a corpus problem rather than something to
                # paper over by padding.
                if scores.shape[0] < tokens:
                    raise ValueError(
                        f"{row.domain}/{row.prompt_id}/{row.rollout_idx}: cache holds "
                        f"{scores.shape[0]} tokens, the label table says {tokens}"
                    )
                records[(layer, lag)].append(
                    {**identity, "scores": scores[:tokens].astype(np.float32)}
                )
        del stack
        if (position + 1) % 250 == 0:
            print(f"  {position + 1}/{len(onset)} rollouts", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument(
        "--activations-root",
        type=Path,
        default=None,
        help="Defaults to <build-root>/activations.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--layers", type=int, nargs="*", default=list(DEFAULT_LAYERS))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-lags", type=int, nargs="*", default=list(DEFAULT_MIN_LAGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    activations_root = args.activations_root or (args.build_root / "activations")
    print(f"Reading layers {args.layers} from {activations_root}", flush=True)
    records = score_split(
        args.build_root,
        activations_root,
        args.split,
        args.layers,
        args.window,
        args.min_lags,
        args.device,
        args.limit,
    )
    for (layer, lag), rows in records.items():
        name = f"actsim_L{layer:02d}_w{args.window}_lag{lag:02d}"
        frame = build_scores(rows)
        path = write_scores(frame, args.out_dir / name / "scores" / f"{args.split}.parquet")
        print(
            f"  {name:<28} {args.split:<22} {len(frame):>6} rollouts "
            f"({int(frame['is_positive'].sum())} positive) -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
