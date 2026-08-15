"""Score every saved checkpoint of a run on the pinned validation rollouts.

A run keeps a checkpoint every fifty steps at every depth, but only ever
measured one of them properly. This produces, for each of those steps and each
depth, the same record that evaluation during training will produce, so a
stopping rule can be applied to a run that has already finished and its answer
means the same thing as it would have meant live.

The whole thing costs about one pass over the rollouts rather than one pass per
checkpoint. With the language model frozen the cached activations are the same
bytes for every checkpoint, and reading them is what a pass costs: a checkpoint
is 12,289 numbers applied to vectors that are already in memory. So the
activations are read once and every checkpoint is applied to them together.

Two identities make that cheap rather than merely possible. Normalising is the
same work for every checkpoint, since only the scale and shift that follow it
are trained, and a scale followed by a linear map is another linear map:

    w . (x_hat * gamma + beta) + b  ==  (w * gamma) . x_hat + (w . beta + b)

so each checkpoint collapses to one vector and one number, and a whole depth's
worth of checkpoints becomes a single matrix multiply against the normalised
states.

    python scripts/replay_checkpoints.py --run-dir outputs/<run>/latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig, LabelConfig
from degeneration_probe.data.activation_store import load_probe_layers
from degeneration_probe.data.dataset import load_degeneration_records
from degeneration_probe.evaluation.head_selection import STEERING_BUDGET, validation_record
from degeneration_probe.evaluation.protocol import WARNING_BANDS

LAYERNORM_EPS = 1e-5


def load_run_config(run_dir: Path) -> ExperimentConfig:
    with (run_dir / "resolved_config.json").open("r", encoding="utf-8") as handle:
        return ExperimentConfig.from_dict(json.load(handle))


def discover_checkpoints(run_dir: Path, every: int) -> List[int]:
    steps = sorted(
        int(path.name.split("-")[1])
        for path in run_dir.glob("checkpoint-*")
        if path.name.split("-")[-1].isdigit()
    )
    return [step for step in steps if step % every == 0]


def collapse_heads(
    run_dir: Path, steps: Sequence[int], layers: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every checkpoint of every depth, as one linear map each.

    Returns weights shaped ``[depths, checkpoints, hidden]`` and biases shaped
    ``[depths, checkpoints]``, already folded through the trained normalization
    so that scoring is a matrix multiply against normalised states.
    """
    weights = torch.zeros(len(layers), len(steps), 4096, dtype=torch.float32)
    biases = torch.zeros(len(layers), len(steps), dtype=torch.float32)
    for column, step in enumerate(steps):
        for row, layer in enumerate(layers):
            head = run_dir / f"checkpoint-{step}" / f"layer_{layer:02d}"
            linear = torch.load(head / "probe.bin", map_location="cpu", weights_only=True)
            w = linear["weight"].reshape(-1).float()
            b = float(linear["bias"].reshape(-1)[0])
            norm_path = head / "normalization.bin"
            if norm_path.is_file():
                norm = torch.load(norm_path, map_location="cpu", weights_only=True)
                gamma = norm["weight"].reshape(-1).float()
                beta = norm["bias"].reshape(-1).float()
                b = b + float(torch.dot(w, beta))
                w = w * gamma
            weights[row, column] = w
            biases[row, column] = b
    return weights, biases


@torch.no_grad()
def score_rollout(
    features: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """One rollout through every checkpoint of every depth.

    ``features`` is ``[depths, tokens, hidden]`` as the cache stores it. The
    result is ``[checkpoints, depths, tokens]``.
    """
    normalised = torch.nn.functional.layer_norm(
        features.float(), (features.shape[-1],), eps=LAYERNORM_EPS
    )
    # [depths, tokens, hidden] x [depths, hidden, checkpoints] -> [depths, tokens, checkpoints]
    logits = torch.bmm(normalised, weights.transpose(1, 2)) + biases.unsqueeze(1)
    return torch.sigmoid(logits).permute(2, 0, 1)


def replay(
    run_dir: Path,
    rollouts: pd.DataFrame,
    *,
    every: int,
    device: torch.device,
    budget: float,
) -> pd.DataFrame:
    config = load_run_config(run_dir)
    layers = list(config.training.probe.probed_layers)
    steps = discover_checkpoints(run_dir, every)
    if not steps:
        raise SystemExit(f"No checkpoints under {run_dir}")
    print(f"{len(steps)} checkpoints x {len(layers)} depths over {len(rollouts)} rollouts")

    weights, biases = collapse_heads(run_dir, steps, layers)
    weights, biases = weights.to(device), biases.to(device)

    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    records = load_degeneration_records(
        config.dataset,
        split=config.dataset.splits.validation,
        label_config=LabelConfig(),
        training=False,
    )
    wanted = set(map(tuple, rollouts[["domain", "prompt_id", "rollout_idx"]].to_numpy()))
    records = [
        r for r in records if (r.domain, r.prompt_id, int(r.rollout_idx)) in wanted
    ]
    if len(records) != len(rollouts):
        raise SystemExit(
            f"{len(records)} of the {len(rollouts)} pinned rollouts were found; the "
            "evaluation population and the dataset disagree."
        )

    # Healthy rollouts contribute one number each, which is all a rollout-level
    # threshold needs. Degenerate rollouts have to keep every token, because
    # coverage is a statement about tokens.
    negative_peaks: List[np.ndarray] = []
    positive_scores: List[np.ndarray] = []
    onsets: List[int] = []

    for index, record in enumerate(records):
        features = load_probe_layers(
            config.dataset.build_root,
            record.domain,
            record.prompt_id,
            record.rollout_idx,
            probe_layers=layers,
        )
        length = min(features.shape[1], len(record.targets))
        scores = score_rollout(features[:, :length].to(device), weights, biases)
        if record.is_positive:
            positive_scores.append(scores.cpu().numpy().astype(np.float32))
            onsets.append(int(record.onset_position))
        else:
            negative_peaks.append(scores.amax(dim=2).cpu().numpy().astype(np.float32))
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(records)} rollouts", flush=True)

    peaks = np.stack(negative_peaks)          # [negatives, checkpoints, depths]
    print(f"{len(positive_scores)} degenerate, {peaks.shape[0]} healthy rollouts")

    rows = []
    for column, step in enumerate(steps):
        for row, layer in enumerate(layers):
            record = validation_record(
                negative_peaks=peaks[:, column, row],
                positive_scores=[s[column, row] for s in positive_scores],
                onsets=onsets,
                budget=budget,
                bands=WARNING_BANDS,
            )
            rows.append({"step": step, "layer": layer, **record})
    return pd.DataFrame(rows), peaks, positive_scores, onsets


def save_threshold_material(path: Path, peaks: np.ndarray, steps, layers) -> None:
    """Keep what the threshold is derived from, so its cost can be reconsidered.

    How many healthy rollouts evaluation reads decides both what a pass costs
    and how firmly the threshold is pinned, and there is no way to argue that
    trade to a number. Every healthy rollout's peak is one float, so keeping all
    of them lets any smaller population be simulated afterwards and the two
    sides of the trade measured rather than assumed.
    """
    np.savez_compressed(
        path, peaks=peaks, steps=np.asarray(steps), layers=np.asarray(layers)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset" / "validation_rollouts_apertus-8b-instruct.csv",
        help="The pinned evaluation population.",
    )
    parser.add_argument(
        "--every", type=int, default=50, help="Replay checkpoints at this step spacing."
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--keep-positive-scores",
        action="store_true",
        help="Also keep every token of every degenerate rollout, which is what "
        "choosing how many of them to evaluate on needs. Gigabytes per run, so "
        "worth it only for the run that choice is made on.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame, peaks, positive_scores, onsets = replay(
        run_dir,
        pd.read_csv(args.rollouts),
        every=args.every,
        device=device,
        budget=STEERING_BUDGET,
    )
    out = args.out or run_dir / "checkpoint_replay.parquet"
    frame.to_parquet(out, index=False)
    print(f"\n{len(frame)} records -> {out}")

    material = run_dir / "checkpoint_replay_thresholds.npz"
    save_threshold_material(material, peaks, frame["step"].unique(), frame["layer"].unique())
    print(f"healthy-rollout peaks {peaks.shape} -> {material}")

    if args.keep_positive_scores:
        # Only worth keeping for the run the evaluation population is chosen on:
        # it is gigabytes per run, against megabytes for the peaks.
        scores_path = run_dir / "checkpoint_replay_positive_scores.npz"
        np.savez_compressed(
            scores_path,
            onsets=np.asarray(onsets),
            **{f"rollout_{i:03d}": s for i, s in enumerate(positive_scores)},
        )
        print(f"degenerate-rollout scores -> {scores_path}")


if __name__ == "__main__":
    main()
