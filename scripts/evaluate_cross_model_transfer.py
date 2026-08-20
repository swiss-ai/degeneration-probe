"""Score a checkpoint against a *different* model's own dataset build.

Reuses score_rollouts.py's per-token scoring machinery unchanged, but swaps
in a different `dataset` config than the one the checkpoint was trained on --
the probe never saw this dataset during training, so this answers whether the
learned head still discriminates degenerating rollouts on a model it was
never fit to. Output lands in the same `scores/<split>.parquet` format
score_rollouts.py writes, so `scripts/evaluate_scores.py` reports it exactly
the way it reports any other run, and the two models' numbers are directly
comparable.

What cannot transfer is a LoRA adapter, not a feature regime. A LoRA run
bakes its weights into the source model's own decoder layers, which mean
nothing in another model's; a plain probe head transfers whether its features
were read from a cache or computed live by running the model.

Both regimes therefore work here. `cached` reads the target build's stored
activations. `adapted` runs the target build's *own* model to produce them,
which is what makes this usable where there is no room to store a cache --
note it is the target's model that gets loaded, not the one the head was
fitted to, since otherwise nothing would have been transferred.

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

from degeneration_probe.config import DatasetConfig, ModelConfig
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.evaluation.scores import write_scores
from degeneration_probe.probes.linear_probe import (
    LiveMultiLayerProbe,
    setup_cached_probe,
    setup_probe,
)
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype
from scripts.score_rollouts import (
    SCORES_DIR,
    load_run_config,
    rollout_metadata,
    score_split,
    score_split_by_layer,
)


def load_dataset_config(path: Path) -> DatasetConfig:
    with path.open("r", encoding="utf-8") as handle:
        return DatasetConfig(**yaml.safe_load(handle))


def target_model_config(dataset_config: DatasetConfig, *, dtype: str) -> ModelConfig:
    """The model whose activations this dataset is made of.

    Named by the build config that generated it, rather than by the run being
    transferred, which names the model the head was fitted to instead. Keeps the
    run's dtype so the only thing that changes is whose representation is read.
    """
    build_path = Path(dataset_config.build_config)
    if not build_path.is_absolute():
        build_path = REPO_ROOT / build_path
    build = DatasetGenConfig.from_yaml(build_path)
    tokenizer_name = getattr(build, "tokenizer_name", None)
    return ModelConfig(
        short_name=dataset_config.short_name,
        name=str(build.model_name),
        tokenizer_name=str(tokenizer_name) if tokenizer_name else None,
        dtype=dtype,
    )


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

    checkpoint_dir = run_dir / checkpoint
    if layer is not None:
        available = config.training.probe.layers
        if available is not None and layer not in available:
            raise ValueError(f"layer {layer} was not trained by this run; it covers {available}")
        checkpoint_dir = checkpoint_dir / f"layer_{layer:02d}"
        config.training.probe.layers = None
        config.training.probe.layer = layer
    # Naming no depth on a multi-depth run scores all of them, filing each under
    # its own layer directory. They share the forward pass, which is the cost.
    all_layers = layer is None and config.training.probe.layers is not None
    if all_layers:
        # A multi-depth checkpoint is a directory of per-depth ones.
        layers = sorted(
            int(d.name.split("_")[1])
            for d in checkpoint_dir.glob("layer_*")
            if (d / "probe_config.json").is_file()
        )
        if not layers:
            raise FileNotFoundError(f"No per-depth probe checkpoints under {checkpoint_dir}")
        config.training.probe.layers = layers
        config.training.probe.layer = layers[0]
    elif not (checkpoint_dir / "probe_config.json").is_file():
        raise FileNotFoundError(f"No probe checkpoint at {checkpoint_dir}")
    # Test the artifact rather than the config: adapter weights are what would
    # actually be loaded onto the target model, and they are fitted to the
    # source model's decoder layers, so they mean nothing there.
    if (checkpoint_dir / "adapter_config.json").is_file():
        raise ValueError(
            f"{checkpoint_dir} carries LoRA adapter weights fitted to "
            f"{config.model.name!r}'s own decoder layers, which have no meaning in "
            "another model's. Only a plain probe head transfers."
        )

    dataset_config = load_dataset_config(dataset_config_path)
    print(
        f"Transferring {checkpoint_dir} (trained on {config.dataset.short_name!r}) "
        f"-> scoring on {dataset_config.short_name!r}"
    )
    config.dataset = dataset_config
    # Evaluation never subsamples, whatever the run was trained with.
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    splits = splits or config.dataset.splits.final_evaluation

    if config.training.features.regime == "cached":
        hidden_size = int(
            list(
                pd.read_parquet(
                    config.dataset.activations_manifest_path,
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
        tokenizer = None
        print(f"Scoring with {checkpoint_dir} (cached features, hidden_size={hidden_size})")
    else:
        # The features must come from the model this dataset is made of. The run
        # config names the model the head was fitted to, which is the wrong one
        # here -- scoring the source model again would transfer nothing.
        config.model = target_model_config(dataset_config, dtype=config.model.dtype)
        model, tokenizer = load_model_and_tokenizer(
            config.model.name,
            tokenizer_name=config.model.tokenizer_name,
            torch_dtype=resolve_torch_dtype(config.model.dtype),
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        if all_layers:
            heads = setup_cached_probe(
                config.training.probe,
                hidden_size=int(model.config.hidden_size),
                checkpoint_path=checkpoint_dir,
                seed=config.training.runtime.seed,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            probe = LiveMultiLayerProbe(model, heads, config.training.probe.layers)
        else:
            _, probe = setup_probe(
                model,
                config.training.probe,
                config.training.lora,
                checkpoint_path=checkpoint_dir,
                seed=config.training.runtime.seed,
            )
        depths = len(config.training.probe.layers) if all_layers else 1
        print(
            f"Scoring with {checkpoint_dir} (adapted features from {config.model.name}, "
            f"{depths} depth(s) per pass)"
        )

    parts = [run_dir, "cross_model", dataset_config.short_name]
    if layer is not None:
        parts.append(f"layer_{layer:02d}")
    base_dir = output_dir or Path(*parts)

    metadata = rollout_metadata(config)
    written = []
    for split in splits:
        if all_layers:
            # One pass, then a file per depth: a score file holds one score per
            # token, so the depths cannot share a file, only the pass.
            frames = score_split_by_layer(
                probe,
                config,
                tokenizer,
                split=split,
                metadata=metadata,
                batch_size=batch_size,
                layers=config.training.probe.layers,
            )
        else:
            frames = {
                layer: score_split(
                    probe, config, tokenizer, split=split, metadata=metadata, batch_size=batch_size
                )
            }
        for depth, frame in sorted(frames.items()):
            target = base_dir / f"layer_{depth:02d}" if all_layers else base_dir
            target.mkdir(parents=True, exist_ok=True)
            path = write_scores(frame, target / SCORES_DIR / f"{split}.parquet")
            tokens = int(frame["num_tokens"].sum())
            positives = int(frame["is_positive"].sum())
            print(
                f"  {split:<22} layer {depth:>2} {len(frame):>6} rollouts "
                f"({positives} positive), {tokens:>10,} tokens -> {path}"
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
