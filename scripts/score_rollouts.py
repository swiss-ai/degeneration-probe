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
from pathlib import Path
from typing import List, Optional

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
from degeneration_probe.training.recording import FINAL_WEIGHTS_DIR, RESOLVED_CONFIG_FILE
from degeneration_probe.utils.file_utils import load_json
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype

SCORES_DIR = "scores"


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
        for row in range(probabilities.shape[0]):
            start = int(batch["prompt_length"][row])
            length = (
                int(batch["target_mask"][row].sum())
                if "features" in batch
                else int(batch["attention_mask"][row].sum()) - start
            )
            scores = probabilities[row, start : start + length]
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
                    "scores": scores.astype(np.float16),
                }
            )
    return build_scores(records)


def run(
    run_dir: Path,
    *,
    checkpoint: str,
    splits: Optional[List[str]],
    batch_size: int,
) -> List[Path]:
    config = load_run_config(run_dir)
    checkpoint_dir = run_dir / checkpoint
    if not (checkpoint_dir / "probe_config.json").is_file():
        raise FileNotFoundError(f"No probe checkpoint at {checkpoint_dir}")

    # Evaluation never subsamples, whatever the run was trained with.
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    splits = splits or config.dataset.splits.final_evaluation

    cached = config.training.features.regime == "cached"
    if cached:
        import pandas as pd_manifest

        hidden_size = int(
            list(
                pd_manifest.read_parquet(
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
        path = write_scores(frame, run_dir / SCORES_DIR / f"{split}.parquet")
        tokens = int(frame["num_tokens"].sum())
        positives = int(frame["is_positive"].sum())
        print(f"  {split:<22} {len(frame):>6} rollouts ({positives} positive), {tokens:>10,} tokens -> {path}")
        written.append(path)
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
    args = parser.parse_args()
    run(
        args.run_dir.resolve(),
        checkpoint=args.checkpoint,
        splits=args.splits,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
