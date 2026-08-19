"""Transplant a LoRA-adapted checkpoint's adapter + probe head onto a *different*
Apertus checkpoint's own weights, and score it against that checkpoint's dataset.

A LoRA adapter (`training.features.regime != "cached"`) bakes its low-rank deltas
into the specific base model it was trained against -- it has no meaning applied
to a different architecture's weights (see evaluate_cross_model_transfer.py,
which is for the opposite case: a plain linear head over frozen activations,
portable to any model's activations but never to a different model's weights).

This script instead asks a narrower question: does the adapter's reshaping of
the residual stream carry over to a *sibling* checkpoint of the same
architecture (same hidden_size, same layer count, same module names) -- e.g.
another Apertus 1.5 SFT variant -- or is it overfit to the exact weight values
of the checkpoint it was trained on? It only ever runs live (no activation
cache is used or produced): the target checkpoint is loaded fresh, the source
adapter + probe head are loaded on top of it via the same setup_probe() path
training itself uses, and every rollout in the target dataset is scored with
one real forward pass -- there is no cheaper way to answer this, since a
different adapter changes the hidden states themselves.

    python scripts/evaluate_lora_transplant.py \
        --run-dir outputs/<lora_run>/<attempt> --checkpoint checkpoint-700 \
        --dataset-config configs/dataset/degeneration-dataset-apertus1p5-sft256k-4200.yaml

    python scripts/evaluate_scores.py --run-dir outputs/<lora_run>/<attempt>/lora_transplant/apertus1p5-sft256k-4200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import DatasetConfig
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.evaluation.scores import write_scores
from degeneration_probe.probes.linear_probe import setup_probe
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype
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
    output_dir: Optional[Path],
) -> List[Path]:
    config = load_run_config(run_dir)
    if config.training.features.regime == "cached":
        raise ValueError(
            f"{run_dir} was trained with training.features.regime='cached' (a plain linear "
            "head over frozen activations, no adapter) -- that transfers to a different "
            "model's activations via evaluate_cross_model_transfer.py instead. This script is "
            "for a LoRA-adapted run's adapter + head, transplanted onto a different "
            "checkpoint's own weights."
        )

    checkpoint_dir = run_dir / checkpoint
    if not (checkpoint_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No LoRA adapter at {checkpoint_dir} (expected adapter_config.json)")

    dataset_config = load_dataset_config(dataset_config_path)
    target_build_config = DatasetGenConfig.from_yaml(Path(dataset_config.build_config))

    print(
        f"Transplanting {checkpoint_dir} (trained on {config.dataset.short_name!r} / "
        f"{config.model.name!r}) onto {target_build_config.model_name!r}, "
        f"scoring on {dataset_config.short_name!r}"
    )
    config.dataset = dataset_config
    # Evaluation never subsamples, whatever the run was trained with.
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    config.model.name = target_build_config.model_name
    config.model.tokenizer_name = target_build_config.tokenizer_name
    splits = splits or config.dataset.splits.final_evaluation

    model, tokenizer = load_model_and_tokenizer(
        config.model.name,
        tokenizer_name=config.model.tokenizer_name,
        torch_dtype=resolve_torch_dtype(config.model.dtype),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    _, probe = setup_probe(
        model,
        config.training.probe,
        config.training.lora,
        checkpoint_path=checkpoint_dir,
        seed=config.training.runtime.seed,
    )
    print(f"Scoring with {checkpoint_dir} (adapter transplanted onto {target_build_config.model_name!r})")

    output_dir = output_dir or (run_dir / "lora_transplant" / dataset_config.short_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = rollout_metadata(config)
    written = []
    for split in splits:
        frame = score_split(
            probe, config, tokenizer, split=split, metadata=metadata, batch_size=batch_size
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
        "--run-dir", type=Path, required=True, help="The LoRA checkpoint's own run directory (an attempt)."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="The sibling checkpoint's training-side dataset config to transplant onto and score against.",
    )
    parser.add_argument("--checkpoint", required=True, help="Which checkpoint inside --run-dir to transplant.")
    parser.add_argument("--splits", nargs="*", default=None, help="Defaults to every evaluation split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the scores go. Defaults to <run-dir>/lora_transplant/<dataset short_name>/.",
    )
    args = parser.parse_args()
    run(
        args.run_dir.resolve(),
        args.dataset_config.resolve(),
        checkpoint=args.checkpoint,
        splits=args.splits,
        batch_size=args.batch_size,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
