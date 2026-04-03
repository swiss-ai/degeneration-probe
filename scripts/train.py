"""Train a degeneration probe on labelled generation data."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from degeneration_probe.config import ProbeConfig, TrainingConfig
from degeneration_probe.dataset import DegenerationDataset, make_collate_fn
from degeneration_probe.model_utils import load_model_and_tokenizer, resolve_torch_dtype
from degeneration_probe.probe import SequenceProbe
from degeneration_probe.training import train
from degeneration_probe.evaluation import evaluate_and_save

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs/train", config_name="main")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Resolve Hydra config into our dataclasses
    probe_cfg = ProbeConfig(
        layer=cfg.probe.layer,
        pooling=cfg.probe.pooling,
        threshold=cfg.probe.threshold,
    )

    # Timestamp-based output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Seed
    torch.manual_seed(cfg.seed)

    # Load model and tokenizer
    log.info("Loading model: %s", cfg.model_name)
    torch_dtype = resolve_torch_dtype(cfg.model_dtype)
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_name, torch_dtype=torch_dtype,
    )

    # Create probe
    log.info("Creating probe on layer %d, pooling=%s", probe_cfg.layer, probe_cfg.pooling)
    probe = SequenceProbe(
        model=model,
        layer_idx=probe_cfg.layer,
        pooling=probe_cfg.pooling,
        seed=cfg.seed,
    )

    # Load dataset
    log.info("Loading training data: %s", cfg.train_data)
    full_dataset = DegenerationDataset(cfg.train_data)
    collate_fn = make_collate_fn(tokenizer, max_length=cfg.max_length)

    # Train/eval split
    if cfg.eval_data is not None:
        train_dataset = full_dataset
        eval_dataset = DegenerationDataset(cfg.eval_data)
    else:
        n_eval = max(1, int(len(full_dataset) * cfg.eval_fraction))
        n_train = len(full_dataset) - n_eval
        train_dataset, eval_dataset = random_split(
            full_dataset,
            [n_train, n_eval],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    log.info("Train: %d samples, Eval: %d samples", len(train_dataset), len(eval_dataset))

    # W&B init
    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"probe_{ts}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Train
    train(
        probe,
        train_loader,
        eval_loader,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        pos_weight=cfg.pos_weight,
        use_wandb=use_wandb,
    )

    # Final evaluation + save
    log.info("Running final evaluation...")
    metrics = evaluate_and_save(
        probe, eval_loader, output_dir / "eval", threshold=probe_cfg.threshold,
    )
    log.info("Final metrics: %s", metrics)

    # Save probe checkpoint
    probe.save(output_dir / "checkpoint")
    log.info("Probe saved to %s", output_dir / "checkpoint")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
