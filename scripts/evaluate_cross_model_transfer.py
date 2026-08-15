"""Score a checkpoint against a *different* model's own dataset build.

Reuses score_rollouts.py's per-token scoring machinery unchanged, but swaps
in a different `dataset` config than the one the checkpoint was trained on --
the probe never saw this dataset during training, so this answers whether the
learned head still discriminates degenerating rollouts on a model it was
never fit to. Output lands in the same `scores/<split>.parquet` format
score_rollouts.py writes, so `scripts/evaluate_scores.py` reports it exactly
the way it reports any other run, and the two models' numbers are directly
comparable.

Only checkpoints trained with `training.features.regime == "cached"` (a
plain linear head over frozen activations, no LoRA adapter) are eligible: a
LoRA-adapted run bakes its adapter weights into the original model's own
decoder layers, which has no meaning against a different base model's
weights or tokenizer.

    python scripts/evaluate_cross_model_transfer.py \
        --run-dir outputs/<apertus_run>/latest \
        --dataset-config configs/dataset/degeneration-dataset-llama3-8b-instruct.yaml \
        --layer 12

    python scripts/evaluate_scores.py --run-dir outputs/<apertus_run>/latest/cross_model/llama3-8b-instruct/layer_12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import DatasetConfig
from degeneration_probe.evaluation.scores import write_scores
from degeneration_probe.probes.linear_probe import setup_cached_probe
from scripts.score_rollouts import SCORES_DIR, load_run_config, rollout_metadata, score_split


def load_dataset_config(path: Path) -> DatasetConfig:
    with path.open("r", encoding="utf-8") as handle:
        return DatasetConfig(**yaml.safe_load(handle))


def run(
    run_dir: Path,
    dataset_config_path: Path,
    *,
    checkpoint: str,
    splits: Optional[List[str]],
    batch_size: int,
    layer: Optional[int],
    output_dir: Optional[Path],
) -> List[Path]:
    config = load_run_config(run_dir)
    if config.training.features.regime != "cached":
        raise ValueError(
            f"{run_dir} was trained with training.features.regime="
            f"{config.training.features.regime!r}, not 'cached'. Only a plain linear head "
            "over frozen activations (no LoRA adapter) is mechanically transferable to a "
            "different base model's activations."
        )

    checkpoint_dir = run_dir / checkpoint
    if layer is not None:
        available = config.training.probe.layers
        if available is not None and layer not in available:
            raise ValueError(f"layer {layer} was not trained by this run; it covers {available}")
        checkpoint_dir = checkpoint_dir / f"layer_{layer:02d}"
        config.training.probe.layers = None
        config.training.probe.layer = layer
    elif config.training.probe.layers is not None:
        raise ValueError(
            f"this run trained {len(config.training.probe.layers)} depths at once. "
            "Name one with --layer, since a scores file holds one score per token "
            "and cannot represent several probes."
        )
    if not (checkpoint_dir / "probe_config.json").is_file():
        raise FileNotFoundError(f"No probe checkpoint at {checkpoint_dir}")

    dataset_config = load_dataset_config(dataset_config_path)
    print(
        f"Transferring {checkpoint_dir} (trained on {config.dataset.short_name!r}) "
        f"-> scoring on {dataset_config.short_name!r}"
    )
    config.dataset = dataset_config
    # Evaluation never subsamples, whatever the run was trained with.
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    splits = splits or config.dataset.splits.final_evaluation

    hidden_size = int(
        list(
            pd.read_parquet(
                Path(config.dataset.build_root) / "activations" / "manifest.parquet",
                columns=["shape"],
            )["shape"].iloc[0]
        )[-1]
    )
    probe = setup_cached_probe(
        config.training.probe,
        hidden_size=hidden_size,
        checkpoint_path=checkpoint_dir,
        seed=config.training.runtime.seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(f"Scoring with {checkpoint_dir} (hidden_size={hidden_size})")

    parts = [run_dir, "cross_model", dataset_config.short_name]
    if layer is not None:
        parts.append(f"layer_{layer:02d}")
    output_dir = output_dir or Path(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = rollout_metadata(config)
    written = []
    for split in splits:
        frame = score_split(
            probe, config, None, split=split, metadata=metadata, batch_size=batch_size
        )
        path = write_scores(frame, output_dir / SCORES_DIR / f"{split}.parquet")
        tokens = int(frame["num_tokens"].sum())
        positives = int(frame["is_positive"].sum())
        print(
            f"  {split:<22} {len(frame):>6} rollouts ({positives} positive), "
            f"{tokens:>10,} tokens -> {path}"
        )
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="The checkpoint's own run directory (an attempt)."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="The other model's training-side dataset config to score against.",
    )
    parser.add_argument("--checkpoint", default="final", help="Which checkpoint inside --run-dir to score with.")
    parser.add_argument("--splits", nargs="*", default=None, help="Defaults to every evaluation split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--layer", type=int, default=None, help="Which depth to score, for a run that trained a head per layer."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the scores go. Defaults to <run-dir>/cross_model/<dataset short_name>/[layer_XX/].",
    )
    args = parser.parse_args()
    run(
        args.run_dir.resolve(),
        args.dataset_config.resolve(),
        checkpoint=args.checkpoint,
        splits=args.splits,
        batch_size=args.batch_size,
        layer=args.layer,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
