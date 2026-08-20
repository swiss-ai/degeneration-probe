"""Score every token of every rollout in a split, and write the result.

This is the only step that needs a GPU. It turns a trained checkpoint into a
table of per-token scores, after which every metric is a small job on a CPU
that can be re-run whenever a question changes.

Scoring is exhaustive by construction: no negative rollout is subsampled and no
rollout is capped, whatever the training configuration said. A cap can only
understate a false-alarm rate, and the false-alarm rate is the number the
deployment decision turns on.

A run directory describes itself, so scoring one needs nothing but its path:

    python scripts/score_rollouts.py --run-dir outputs/<run_name>/<attempt>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.data.dataset import (
    create_degeneration_dataset,
    degeneration_collate_fn,
)
from degeneration_probe.evaluation.scores import build_scores, write_scores
from degeneration_probe.data.cached_dataset import cached_collate_fn, create_cached_dataset
from degeneration_probe.probes.linear_probe import setup_cached_probe, setup_probe
from degeneration_probe.training.recording import (
    FINAL_WEIGHTS_DIR,
    RESOLVED_CONFIG_FILE,
    environment_info,
)
from degeneration_probe.utils.file_utils import load_json, save_json
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype

SCORES_DIR = "scores"
# Which weights produced the scores in a directory. Without it a scores file is
# anonymous: the thresholds frozen beside it describe some scorer, and nothing
# says which. Recovering that afterwards is not generally possible, since
# neighbouring checkpoints of one depth differ by less than the precision the
# stored metrics carry.
PROVENANCE_FILE = "scoring_provenance.json"


def check_provenance(output_dir: Path, checkpoint: str, layer: Optional[int]) -> None:
    """Refuse to add a second scorer's output to one directory.

    A directory of scores is read as one scorer: the thresholds frozen on its
    validation split are applied to every other split beside it. Scoring one
    split with one checkpoint and another split with a different one leaves a
    directory that no threshold describes, and the resulting report looks
    perfectly ordinary. Naming a separate destination is the way to hold two
    checkpoints at once.
    """
    path = output_dir / PROVENANCE_FILE
    if not path.is_file():
        return
    recorded = load_json(path)
    same = recorded.get("checkpoint") == checkpoint and recorded.get("layer") == layer
    if same:
        return
    raise SystemExit(
        f"{output_dir} already holds scores from checkpoint "
        f"{recorded.get('checkpoint')!r} at layer {recorded.get('layer')!r}, and this "
        f"call would add checkpoint {checkpoint!r} at layer {layer!r} beside them. "
        "Thresholds frozen in a directory describe the scorer that filled it, so the "
        "two cannot share one. Pass --output-dir to give this checkpoint its own."
    )


def write_provenance(
    output_dir: Path,
    *,
    run_dir: Path,
    checkpoint: str,
    checkpoint_dir: Path,
    layer: Optional[int],
    regime: str,
    splits: List[str],
) -> Path:
    """Record which weights filled this directory, and when."""
    path = output_dir / PROVENANCE_FILE
    previous = load_json(path).get("splits", []) if path.is_file() else []
    environment = environment_info()
    save_json(
        {
            "run_dir": str(run_dir),
            "checkpoint": checkpoint,
            "checkpoint_dir": str(checkpoint_dir),
            "layer": layer,
            "regime": regime,
            "splits": sorted(set(previous) | set(splits)),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
        },
        path,
    )
    return path


def load_run_config(run_dir: Path) -> ExperimentConfig:
    config_path = run_dir / RESOLVED_CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"{run_dir} does not look like a run directory: no {RESOLVED_CONFIG_FILE}")
    return ExperimentConfig.from_dict(load_json(config_path))


def rollout_metadata(config: ExperimentConfig) -> pd.DataFrame:
    """The frontier and stop reason for every rollout, read once."""
    path = Path(config.dataset.build_root) / "onset_labels" / "onset_labels.parquet"
    columns = ["prompt_id", "rollout_idx", "stop_reason", "onset_position", "is_positive"]
    return pd.read_parquet(path, columns=columns)


@torch.no_grad()
def _score_records(
    probe,
    config: ExperimentConfig,
    tokenizer,
    *,
    split: str,
    metadata: pd.DataFrame,
    batch_size: int,
) -> List[dict]:
    """One record per rollout, each carrying ``[tokens, depths]`` of scores."""
    if tokenizer is None:
        dataset = create_cached_dataset(
            config.dataset,
            split=split,
            label_config=config.training.label,
            probe_layer=config.training.probe.layer,
            training=False,
        )
        collate = cached_collate_fn
    else:
        dataset = create_degeneration_dataset(
            config.dataset,
            tokenizer,
            split=split,
            label_config=config.training.label,
            training=False,
        )
        collate = degeneration_collate_fn
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.training.runtime.dataloader_num_workers,
    )
    probe.eval()
    device = getattr(probe, "device", next(probe.parameters()).device)
    lookup = metadata.set_index(["prompt_id", "rollout_idx"])

    records = []
    for batch in loader:
        model_inputs = (
            {"features": batch["features"].to(device)}
            if "features" in batch
            else {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
        )
        logits = probe(**model_inputs)["probe_logits"]
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        # A probe over several depths returns one logit per depth from the same
        # pass, so the trailing axis indexes the head rather than the token.
        # Squeezed away for one depth, which keeps every single-depth caller
        # reading the same two-dimensional array it always did.
        if probabilities.ndim == 2:
            probabilities = probabilities[..., None]
        for row in range(probabilities.shape[0]):
            start = int(batch["prompt_length"][row])
            length = (
                int(batch["target_mask"][row].sum())
                if "features" in batch
                else int(batch["attention_mask"][row].sum()) - start
            )
            scores = probabilities[row, start : start + length, :]
            prompt_id = batch["prompt_id"][row]
            rollout_idx = int(batch["rollout_idx"][row])
            labels = lookup.loc[(prompt_id, rollout_idx)]
            onset = labels["onset_position"]
            records.append(
                {
                    "prompt_id": prompt_id,
                    "rollout_idx": rollout_idx,
                    "domain": batch["domain"][row],
                    "split": batch["split"][row],
                    "stop_reason": str(labels["stop_reason"]),
                    # The scored length, which is the rollout truncated to the
                    # configured completion budget if it ever exceeded it.
                    "num_tokens": int(length),
                    "onset_position": float(onset) if pd.notna(onset) else None,
                    "is_positive": bool(labels["is_positive"]) and pd.notna(onset),
                    "scores": scores.astype(np.float32),
                }
            )
    return records


def score_split(
    probe,
    config: ExperimentConfig,
    tokenizer,
    *,
    split: str,
    metadata: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    """Run the probe over one split and return its score table."""
    records = _score_records(
        probe, config, tokenizer, split=split, metadata=metadata, batch_size=batch_size
    )
    for record in records:
        record["scores"] = record["scores"][:, 0]
    return build_scores(records)


def score_split_by_layer(
    probe,
    config: ExperimentConfig,
    tokenizer,
    *,
    split: str,
    metadata: pd.DataFrame,
    batch_size: int,
    layers: Sequence[int],
) -> Dict[int, pd.DataFrame]:
    """The same split scored at several depths, from a single pass over it.

    A score file holds one score per token, so the depths cannot share one; what
    they can share is the forward pass that produced them, which is the whole
    cost. Scoring each depth separately would recompute an identical pass per
    depth and discard all but one column of it.
    """
    records = _score_records(
        probe, config, tokenizer, split=split, metadata=metadata, batch_size=batch_size
    )
    width = records[0]["scores"].shape[-1] if records else len(layers)
    if width != len(layers):
        raise ValueError(
            f"the probe returned {width} depths but {len(layers)} were named; "
            "the scores would be filed under the wrong layers"
        )
    frames = {}
    for index, layer in enumerate(layers):
        frames[int(layer)] = build_scores(
            [{**record, "scores": record["scores"][:, index]} for record in records]
        )
    return frames


def run(
    run_dir: Path,
    *,
    checkpoint: str,
    splits: Optional[List[str]],
    batch_size: int,
    layer: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    config = load_run_config(run_dir)
    checkpoint_dir = run_dir / checkpoint
    requested_output = output_dir
    output_dir = run_dir

    if layer is not None:
        # A run covering several depths writes each head as an ordinary
        # single-layer probe, so one depth is scored by pointing the ordinary
        # path at that head. Scores go to their own directory, since the whole
        # reason to score several depths is to compare them and a shared one
        # would leave only the last.
        available = config.training.probe.probed_layers
        if layer not in available:
            raise ValueError(
                f"layer {layer} was not trained by this run; it covers {available}"
            )
        checkpoint_dir = checkpoint_dir / f"layer_{layer:02d}"
        config.training.probe.layers = None
        config.training.probe.layer = layer
        output_dir = run_dir / "layers" / f"layer_{layer:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
    elif config.training.probe.layers is not None:
        raise ValueError(
            f"this run trained {len(config.training.probe.layers)} depths at once. "
            "Name one with --layer, since a scores file holds one score per token "
            "and cannot represent several probes."
        )

    if requested_output is not None:
        # Scoring writes to one place per run and depth, so scoring a second
        # checkpoint of the same depth would replace the first. Naming a
        # destination is what lets several checkpoints of one run be compared,
        # which is how a metric's trajectory over training is measured.
        output_dir = requested_output
        output_dir.mkdir(parents=True, exist_ok=True)

    if not (checkpoint_dir / "probe_config.json").is_file():
        raise FileNotFoundError(f"No probe checkpoint at {checkpoint_dir}")

    # Before the GPU is touched, so a mismatch costs nothing to discover.
    check_provenance(output_dir, checkpoint, layer)

    # Evaluation never subsamples, whatever the run was trained with.
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    splits = splits or config.dataset.splits.final_evaluation

    cached = config.training.features.regime == "cached"
    if cached:
        import pandas as pd_manifest

        hidden_size = int(
            list(
                pd_manifest.read_parquet(
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
    else:
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
    print(f"Scoring with {checkpoint_dir} ({config.training.features.regime} features)")

    metadata = rollout_metadata(config)
    written = []
    for split in splits:
        frame = score_split(
            probe,
            config,
            tokenizer,
            split=split,
            metadata=metadata,
            batch_size=batch_size,
        )
        path = write_scores(frame, output_dir / SCORES_DIR / f"{split}.parquet")
        tokens = int(frame["num_tokens"].sum())
        positives = int(frame["is_positive"].sum())
        print(f"  {split:<22} {len(frame):>6} rollouts ({positives} positive), {tokens:>10,} tokens -> {path}")
        written.append(path)

    record = write_provenance(
        output_dir,
        run_dir=run_dir,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        layer=layer,
        regime=config.training.features.regime,
        splits=list(splits),
    )
    print(f"  provenance -> {record}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="A run directory (an attempt).")
    parser.add_argument(
        "--checkpoint",
        default=FINAL_WEIGHTS_DIR,
        help="Which checkpoint inside the run directory to score with (default: final).",
    )
    parser.add_argument("--splits", nargs="*", default=None, help="Defaults to every evaluation split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Which depth to score, for a run that trained a head per layer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the scores go. Needed to keep several checkpoints of one run side by side.",
    )
    args = parser.parse_args()
    run(
        args.run_dir.resolve(),
        checkpoint=args.checkpoint,
        splits=args.splits,
        batch_size=args.batch_size,
        layer=args.layer,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
