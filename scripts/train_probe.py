"""Train the scalar degeneration probe from the local materialized dataset."""

from __future__ import annotations

import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.data.cached_dataset import cached_collate_fn, create_cached_dataset
from degeneration_probe.data.windowed_dataset import WindowedActivationDataset
from degeneration_probe.data.dataset import (
    compute_pos_weight,
    create_degeneration_dataset,
    degeneration_collate_fn,
)
from degeneration_probe.evaluation.metrics import build_validation_metrics
from degeneration_probe.probes.linear_probe import setup_cached_probe, setup_probe
from degeneration_probe.training.arguments import build_training_arguments
from degeneration_probe.training.recording import (
    DATASET_SUMMARY_FILE,
    FINAL_EVALUATION_FILE,
    FINAL_WEIGHTS_DIR,
    RESOLVED_CONFIG_FILE,
    CollapseGuard,
    ResampleWindows,
    RunInfo,
    RunRecorder,
    prepare_run_location,
)
from degeneration_probe.training.run_identity import (
    derive_group_name,
    derive_run_name,
    derive_tags,
    run_axes,
)
from degeneration_probe.training.trainer import ProbeTrainer
from degeneration_probe.utils.file_utils import save_json
from degeneration_probe.utils.model_utils import (
    load_model_and_tokenizer,
    print_trainable_parameters,
    resolve_torch_dtype,
)


def activation_hidden_size(config: ExperimentConfig) -> int:
    """Read the width of the cached states from the cache itself."""
    import pandas as pd

    manifest = pd.read_parquet(
        Path(config.dataset.build_root) / "activations" / "manifest.parquet",
        columns=["shape"],
    )
    return int(list(manifest["shape"].iloc[0])[-1])


def _trainable_parameter_counts(probe) -> dict:
    probe_parameters = sum(
        parameter.numel()
        for name, parameter in probe.named_parameters()
        if parameter.requires_grad and not name.startswith("model.")
    )
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in probe.named_parameters()
        if parameter.requires_grad and name.startswith("model.")
    )
    return {
        "probe_parameters": probe_parameters,
        "adapter_parameters": adapter_parameters,
        "trainable_parameters": probe_parameters + adapter_parameters,
    }


def run(config: ExperimentConfig, resolved_config: dict) -> dict:
    load_dotenv()
    # Validate extension names before allocating the language model.
    build_validation_metrics(config.training.validation.metrics)

    run_name = derive_run_name(config)
    location = prepare_run_location(
        Path(to_absolute_path(config.training.checkpoint.root_dir)),
        run_name,
        config.training.checkpoint.resume,
    )
    run_dir = location.run_dir
    resume_from = location.resume_from

    axes = run_axes(config)
    tags = derive_tags(config)
    run_info = RunInfo(
        run_name=run_name,
        group=derive_group_name(config),
        tags=tags,
        axes=axes,
        run_dir=str(run_dir),
        attempt=location.attempt,
    )
    run_info.training["resumed_from"] = resume_from
    run_info.write(run_dir)
    save_json(resolved_config, run_dir / RESOLVED_CONFIG_FILE)

    tracking_enabled = config.training.wandb.enabled and config.training.wandb.mode != "disabled"
    wandb_run = None
    if tracking_enabled:
        import wandb

        previous_id = location.wandb_id if config.training.wandb.resume else None
        wandb_run = wandb.init(
            entity=config.training.wandb.entity,
            project=config.training.wandb.project,
            name=run_name,
            group=run_info.group,
            job_type=config.training.wandb.job_type,
            tags=tags,
            notes=config.training.wandb.notes,
            mode=config.training.wandb.mode,
            id=previous_id,
            resume="allow" if previous_id else None,
            config={
                **resolved_config,
                "axes": axes,
                "environment": run_info.environment,
                "run_dir": str(run_dir),
            },
        )
        run_info.wandb = {
            "id": wandb_run.id,
            "url": wandb_run.url,
            "project": wandb_run.project,
            "entity": wandb_run.entity,
            "group": run_info.group,
        }
        run_info.write(run_dir)

    print(f"Run name: {run_name}")
    print(f"Run directory: {run_dir}" + (" (continuing)" if location.continued else ""))
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    print("Resolved config:")
    print(OmegaConf.to_yaml(OmegaConf.create(resolved_config), resolve=True))

    try:
        return _train(
            config,
            run_dir=run_dir,
            run_name=run_name,
            run_info=run_info,
            resume_from=resume_from,
            wandb_run=wandb_run,
        )
    except Exception as error:  # noqa: BLE001 - the status has to survive the failure
        run_info.finish(run_dir, status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()


def _train(
    config: ExperimentConfig,
    *,
    run_dir: Path,
    run_name: str,
    run_info: RunInfo,
    resume_from,
    wandb_run,
) -> dict:
    # The cached regime never allocates the language model: that is the point of
    # it, and it is what makes a sweep over recipes affordable.
    cached = config.training.features.regime == "cached"
    if cached:
        import torch

        probe = setup_cached_probe(
            config.training.probe,
            hidden_size=activation_hidden_size(config),
            seed=config.training.runtime.seed,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        collate = cached_collate_fn

        def build(split, training):
            dataset = create_cached_dataset(
                config.dataset,
                split=split,
                label_config=config.training.label,
                probe_layer=config.training.probe.layer,
                training=training,
            )
            if not training:
                # Evaluation reads whole rollouts: a selection rule describes
                # what a run learns from, never what it is measured on.
                return dataset
            return WindowedActivationDataset(
                dataset.records,
                build_root=config.dataset.build_root,
                probe_layer=config.training.probe.layer,
                selection=config.training.selection,
                batch_size=config.training.runtime.per_device_train_batch_size,
                seed=config.training.runtime.seed,
            )
    else:
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
        collate = degeneration_collate_fn

        def build(split, training):
            return create_degeneration_dataset(
                config.dataset,
                tokenizer,
                split=split,
                label_config=config.training.label,
                training=training,
            )

    print_trainable_parameters(probe)

    loss_name = config.training.loss.name
    splits = config.dataset.splits
    datasets = {splits.train: build(splits.train, True)}
    for split in splits.final_evaluation:
        datasets[split] = build(split, False)

    dataset_summary = {split: dataset.summary() for split, dataset in datasets.items()}
    pos_weight = None
    if loss_name == "bce":
        pos_weight = (
            compute_pos_weight(datasets[splits.train])
            if config.training.loss.bce.use_pos_weight
            else 1.0
        )
        train_summary = dataset_summary[splits.train]
        negative_tokens = train_summary["negative_tokens"]
        # Subsampling the population and weighting the loss correct the same
        # skew. This ratio is what says the correction happened once: it lands
        # on one when the weight matches the sampled population, and shows the
        # residual skew when the weight is switched off.
        dataset_summary["pos_weight"] = pos_weight
        dataset_summary["effective_positive_ratio"] = (
            pos_weight * train_summary["positive_tokens"] / negative_tokens
            if negative_tokens
            else None
        )
        print(f"Training BCE pos_weight: {pos_weight:.6g}")
    save_json(dataset_summary, run_dir / DATASET_SUMMARY_FILE)
    run_info.training["dataset_summary"] = dataset_summary
    run_info.training.update(_trainable_parameter_counts(probe))
    run_info.write(run_dir)
    if wandb_run is not None:
        wandb_run.config.update(
            {"data": dataset_summary, **_trainable_parameter_counts(probe)},
            allow_val_change=True,
        )

    tokens_per_step = config.training.budget.tokens_per_step
    if tokens_per_step is not None:
        window = getattr(config.training.selection, "window_size", 1)
        per_batch = config.training.runtime.per_device_train_batch_size * window
        accumulation = max(1, round(tokens_per_step / per_batch))
        config.training.runtime.gradient_accumulation_steps = accumulation
        realized = accumulation * per_batch
        run_info.training["budget"] = {
            "tokens_per_step_requested": tokens_per_step,
            "tokens_per_step_realized": realized,
            "gradient_accumulation_steps": accumulation,
        }
        print(f"Budget: {realized} tokens per step over {accumulation} accumulated batches")

    checkpoint = config.training.checkpoint
    training_args = build_training_arguments(
        config,
        run_dir=run_dir,
        run_name=run_name,
        report_to=["wandb"] if wandb_run is not None else [],
    )

    recorder = RunRecorder(run_dir)
    callbacks = [recorder, CollapseGuard(config.training.budget.collapse_threshold)]
    if config.training.selection.resample_each_epoch and hasattr(
        datasets[splits.train], "resample"
    ):
        callbacks.append(ResampleWindows(datasets[splits.train]))
    if config.training.budget.patience:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.training.budget.patience)
        )
    trainer = ProbeTrainer(
        model=probe,
        cfg=config.training,
        args=training_args,
        train_dataset=datasets[splits.train],
        eval_dataset=datasets[splits.validation],
        validation_dataset=datasets[splits.validation],
        final_evaluation_datasets={
            split: datasets[split] for split in splits.final_evaluation
        },
        pos_weight=pos_weight,
        data_collator=collate,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    final_dir = run_dir / FINAL_WEIGHTS_DIR
    trainer.save_model(str(final_dir))
    save_json(asdict(config), final_dir / "training_config.json")

    final_metrics = trainer.evaluate_final()
    save_json(final_metrics, run_dir / FINAL_EVALUATION_FILE)

    run_info.training.update(
        {
            "global_step": trainer.state.global_step,
            "epochs_completed": trainer.state.epoch,
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "metric_for_best_model": checkpoint.metric_for_best_model,
            "train_runtime_seconds": train_result.metrics.get("train_runtime"),
            "final_weights": str(final_dir),
            "final_metrics": final_metrics,
        }
    )
    run_info.finish(run_dir, status="finished")
    recorder.write_history()

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                **final_metrics,
                "best_metric": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "global_step": trainer.state.global_step,
                "run_dir": str(run_dir),
            }
        )
    return final_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def hydra_entry(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    experiment = ExperimentConfig.from_dict(resolved)
    run(experiment, resolved)


if __name__ == "__main__":
    hydra_entry()
