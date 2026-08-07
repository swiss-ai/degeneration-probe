"""Train the scalar degeneration probe from the local materialized dataset."""

from __future__ import annotations

import atexit
import sys
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import TrainerCallback, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.data.dataset import (
    compute_pos_weight,
    create_degeneration_dataset,
    degeneration_collate_fn,
)
from degeneration_probe.evaluation.metrics import build_validation_metrics
from degeneration_probe.probes.linear_probe import setup_probe
from degeneration_probe.training.trainer import ProbeTrainer
from degeneration_probe.utils.file_utils import save_json
from degeneration_probe.utils.model_utils import (
    load_model_and_tokenizer,
    print_trainable_parameters,
    resolve_torch_dtype,
)


def run(config: ExperimentConfig, resolved_config: dict) -> dict:
    load_dotenv()
    # Validate extension names before allocating the language model.
    build_validation_metrics(config.training.validation.metrics)
    output_dir = Path(to_absolute_path(config.training.checkpoint.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(resolved_config, output_dir / "resolved_config.json")

    tracking_enabled = config.training.wandb.enabled and config.training.wandb.mode != "disabled"
    if tracking_enabled:
        import wandb

        wandb.init(
            project=config.training.wandb.project,
            name=config.training.wandb.name,
            tags=config.training.wandb.tags or None,
            mode=config.training.wandb.mode,
            config=resolved_config,
        )

    print("Resolved config:")
    print(OmegaConf.to_yaml(OmegaConf.create(resolved_config), resolve=True))
    model, tokenizer = load_model_and_tokenizer(
        config.model.name,
        tokenizer_name=config.model.tokenizer_name,
        torch_dtype=resolve_torch_dtype(config.model.dtype),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if config.training.runtime.gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

    _, probe = setup_probe(
        model,
        config.training.probe,
        config.training.lora,
        seed=config.training.runtime.seed,
    )
    print_trainable_parameters(probe)

    loss_name = config.training.loss.name
    splits = config.dataset.splits
    train_dataset = create_degeneration_dataset(
        config.dataset,
        tokenizer,
        split=splits.train,
        loss_name=loss_name,
        training=True,
    )
    validation_dataset = create_degeneration_dataset(
        config.dataset,
        tokenizer,
        split=splits.validation,
        loss_name=loss_name,
        training=False,
    )
    test_indomain_dataset = create_degeneration_dataset(
        config.dataset,
        tokenizer,
        split=splits.test_indomain,
        loss_name=loss_name,
        training=False,
    )
    test_heldout_dataset = create_degeneration_dataset(
        config.dataset,
        tokenizer,
        split=splits.test_heldout_domains,
        loss_name=loss_name,
        training=False,
    )
    pos_weight = None
    if loss_name == "bce":
        pos_weight = (
            compute_pos_weight(train_dataset)
            if config.training.loss.bce.use_pos_weight
            else 1.0
        )
        print(f"Training BCE pos_weight: {pos_weight:.6g}")

    runtime = config.training.runtime
    validation = config.training.validation
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=runtime.per_device_train_batch_size,
        per_device_eval_batch_size=runtime.per_device_eval_batch_size,
        gradient_accumulation_steps=runtime.gradient_accumulation_steps,
        num_train_epochs=runtime.num_train_epochs,
        max_steps=runtime.max_steps,
        max_grad_norm=runtime.max_grad_norm,
        logging_steps=runtime.logging_steps,
        dataloader_num_workers=runtime.dataloader_num_workers,
        eval_strategy=validation.strategy,
        eval_steps=validation.steps,
        save_strategy="no",
        remove_unused_columns=False,
        label_names=["targets"],
        report_to=["wandb"] if tracking_enabled else [],
        run_name=config.training.wandb.name,
        learning_rate=config.training.optimizer.probe_learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        seed=runtime.seed,
        logging_first_step=True,
    )

    trainer = ProbeTrainer(
        model=probe,
        cfg=config.training,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        validation_dataset=validation_dataset,
        final_evaluation_datasets={
            splits.validation: validation_dataset,
            splits.test_indomain: test_indomain_dataset,
            splits.test_heldout_domains: test_heldout_dataset,
        },
        pos_weight=pos_weight,
        data_collator=degeneration_collate_fn,
    )

    def save_checkpoint(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        probe.save(path)
        tokenizer.save_pretrained(path)
        save_json(asdict(config), path / "training_config.json")

    class EpochCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if config.training.checkpoint.save_each_epoch:
                epoch = int(round(state.epoch or 0))
                save_checkpoint(output_dir / f"epoch_{epoch}")

    trainer.add_callback(EpochCheckpointCallback())
    atexit.register(save_checkpoint, output_dir)
    trainer.train()
    save_checkpoint(output_dir)
    final_metrics = trainer.evaluate_final()
    save_json(final_metrics, output_dir / "final_evaluation.json")
    atexit.unregister(save_checkpoint)

    if tracking_enabled:
        import wandb

        wandb.finish()
    return final_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def hydra_entry(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    experiment = ExperimentConfig.from_dict(resolved)
    run(experiment, resolved)


if __name__ == "__main__":
    hydra_entry()
