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
    load_token_signal,
)
from degeneration_probe.evaluation.metrics import build_validation_metrics
from degeneration_probe.probes.linear_probe import setup_cached_probe, setup_probe
from degeneration_probe.training.arguments import (
    build_training_arguments,
    resolve_token_budget,
)
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


MONITOR_SEED = 0


def subsample_for_monitoring(dataset, max_rollouts, *, seed: int = MONITOR_SEED):
    """Cut the monitoring split down, keeping every positive rollout.

    Positives are the scarce class and the rank metric is built from them, so
    dropping any would make the curve noisier for no saving worth having. The
    negatives are thinned instead.

    The choice is fixed rather than drawn from the run's own seed, so that seed
    repeats of one recipe are monitored on identical rows. Letting the monitor
    move with the seed would add a difference in what was measured to the
    difference the seeds exist to measure.
    """
    if max_rollouts is None or max_rollouts >= len(dataset.records):
        return dataset
    import copy

    import numpy as np

    positives = [r for r in dataset.records if r.is_positive]
    negatives = [r for r in dataset.records if not r.is_positive]
    room = max(0, max_rollouts - len(positives))
    if room and len(negatives) > room:
        chosen = np.random.default_rng(seed).choice(len(negatives), size=room, replace=False)
        negatives = [negatives[index] for index in sorted(chosen)]
    reduced = copy.copy(dataset)
    reduced.records = positives + negatives
    return reduced


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
                probe_layer=config.training.probe.probed_layers,
                training=training,
            )
            if not training:
                # Evaluation reads whole rollouts: a selection rule describes
                # what a run learns from, never what it is measured on.
                return dataset
            hardness = None
            if config.training.selection.strategy == "frontier_window_hard_negative":
                # Hard-negative placement needs to know where a rollout looks
                # repetitive, which only the negatives are placed by.
                negatives = [
                    (index, record)
                    for index, record in enumerate(dataset.records)
                    if not record.is_positive
                ]
                signal = load_token_signal(
                    config.dataset, [record for _, record in negatives]
                )
                hardness = {
                    negatives[position][0]: values
                    for position, values in signal.items()
                }
            return WindowedActivationDataset(
                dataset.records,
                build_root=config.dataset.build_root,
                probe_layer=config.training.probe.probed_layers,
                selection=config.training.selection,
                batch_size=config.training.runtime.per_device_train_batch_size,
                seed=config.training.runtime.seed,
                hardness=hardness,
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
                # A rule applied while the model adapts masks the loss rather
                # than shrinking the batch, since the prefix must run anyway.
                selection=config.training.selection
                if training and config.training.selection.strategy != "all_tokens"
                else None,
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
        budget = resolve_token_budget(
            tokens_per_step,
            valid_tokens=dataset_summary[splits.train]["valid_tokens"],
            examples=len(datasets[splits.train]),
            per_device_batch_size=config.training.runtime.per_device_train_batch_size,
        )
        config.training.runtime.gradient_accumulation_steps = budget[
            "gradient_accumulation_steps"
        ]
        run_info.training["budget"] = budget
        print(
            f"Budget: {budget['tokens_per_step_realized']} tokens per step over "
            f"{budget['gradient_accumulation_steps']} accumulated batches "
            f"({budget['tokens_per_example']:.1f} tokens per example)"
        )

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
    monitor = subsample_for_monitoring(
        datasets[splits.validation], config.training.validation.max_rollouts
    )
    # An empty list means none: scoring now happens from saved checkpoints, so
    # an exploratory run has no reason to spend an hour evaluating splits it
    # will not read. Only an unset value falls back to every split.
    configured = config.training.validation.final_splits
    final_splits = splits.final_evaluation if configured is None else list(configured)
    unknown = set(final_splits) - set(datasets)
    if unknown:
        raise ValueError(
            f"training.validation.final_splits names {sorted(unknown)}, which the dataset "
            f"does not define. Available: {sorted(datasets)}"
        )
    if monitor is not datasets[splits.validation]:
        print(
            f"Monitoring on {len(monitor)} of {len(datasets[splits.validation])} "
            f"{splits.validation} rollouts, every positive kept"
        )
    trainer = ProbeTrainer(
        model=probe,
        cfg=config.training,
        args=training_args,
        train_dataset=datasets[splits.train],
        eval_dataset=monitor,
        validation_dataset=monitor,
        final_evaluation_datasets={split: datasets[split] for split in final_splits},
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
