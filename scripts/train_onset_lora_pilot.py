"""Subtasks I and J: LoRA-condition training for both onset probe designs, multi-horizon
and probe_N. `--horizons` controls which: pass all 5 (the default) for the multi-horizon
design (subtask I), or a single horizon (e.g. `--horizons 5`) for one probe_N run
(subtask J) -- both go through the exact same "onset_multi_horizon" task machinery,
since `compute_multi_horizon_bce_loss` and friends (subtask H) are already generic over
any horizons list length, including 1. One script instead of two avoids duplicating the
training loop. (The implementation prompt's original sequencing recommendation was to
pilot multi-horizon first and gate probe_N x LoRA on its result, since probe_N needs K
separate full fine-tunes vs. multi-horizon's one -- the project owner explicitly chose
to run both overnight without waiting on that gate, 2026-07-13.)

Data: subtask G's `select_lora_pilot_rollouts` / `build_onset_probing_items`
(`degeneration_probe/data/onset_lora_dataset.py`) -- all positive rollouts in each of
`train`/`val`, plus a domain-stratified negative sample sized to that split's share of
the implementation prompt's recommended 2,000-3,000-negative total.

Hyperparameters: mirror `configs/training/apertus_stability_fp32.yaml` as closely as
possible per the project owner's explicit request (2026-07-13) -- that config was
tuned for Apertus-8B hallucination-probe LoRA stability: `model_dtype=bfloat16`,
`lora_layers="all"`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`,
`probe_dtype="float32"` (probe head runs in fp32 for stability even though the LM
stays bf16), `normalize_before_head="layernorm"`, `probe_head_lr=lora_lr=1e-4`.

This is NOT run through the Hydra CLI (`scripts/train_probe.py`) since dataset
construction is entirely different here (local parquet + subtask G's item builder, not
`create_probing_dataset`'s HF-Hub path) -- everything else (TrainingConfig, ProbeTrainer,
TokenizedProbingDataset, the collate fn) is reused as-is.
"""

from __future__ import annotations

import argparse
import atexit
from pathlib import Path
from typing import List

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import TrainingArguments

from degeneration_probe.config import ProbeConfig, TrainingConfig
from degeneration_probe.data.dataset import TokenizedProbingDataset, TokenizedProbingDatasetConfig, tokenized_probing_collate_fn
from degeneration_probe.data.onset_dataset import build_rollout_index
from degeneration_probe.data.onset_lora_dataset import build_onset_probing_items, select_lora_pilot_rollouts
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.onset_labels import DEFAULT_HORIZONS, eligible_token_counts_per_horizon
from degeneration_probe.probes.value_head_probe import setup_probe
from degeneration_probe.training.trainer import ProbeTrainer
from degeneration_probe.utils.file_utils import save_json
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, print_trainable_parameters, resolve_torch_dtype

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"

# Implementation prompt's recommended total pilot negative-rollout budget (2,000-3,000);
# split proportionally to each split's share of the 862 corpus-wide positive rollouts.
TOTAL_NEGATIVE_TARGET = 2500


def horizon_weights_for(rollout_index: pd.DataFrame, horizons: List[int]) -> List[float]:
    """Inverse-positive-rate weight per horizon, same convention
    scripts/train_onset_probes.py uses for the no-LoRA condition."""
    counts_df = eligible_token_counts_per_horizon(rollout_index, horizons=horizons)
    return [
        float((row.eligible_tokens - row.positive_tokens) / row.positive_tokens)
        for row in counts_df.itertuples(index=False)
    ]


def build_pilot_datasets(
    dataset_config: DatasetGenConfig,
    horizons: List[int],
    max_length: int,
    max_completion_length: int,
    seed: int,
    smoke_test_n: int = 0,
) -> tuple:
    onset_labels_df = pd.read_parquet(paths.onset_labels_path(dataset_config))
    train_index = build_rollout_index(onset_labels_df, "train")
    val_index = build_rollout_index(onset_labels_df, "val")

    n_train_pos = int(train_index["is_positive"].sum())
    n_val_pos = int(val_index["is_positive"].sum())
    n_total_pos = n_train_pos + n_val_pos
    train_neg_target = round(TOTAL_NEGATIVE_TARGET * n_train_pos / n_total_pos)
    val_neg_target = round(TOTAL_NEGATIVE_TARGET * n_val_pos / n_total_pos)

    selected_train = select_lora_pilot_rollouts(train_index, n_negative_target=train_neg_target, seed=seed)
    selected_val = select_lora_pilot_rollouts(val_index, n_negative_target=val_neg_target, seed=seed)

    if smoke_test_n > 0:
        selected_train = pd.concat(
            [
                selected_train[selected_train.is_positive].head(smoke_test_n),
                selected_train[~selected_train.is_positive].head(smoke_test_n),
            ]
        ).reset_index(drop=True)
        selected_val = pd.concat(
            [
                selected_val[selected_val.is_positive].head(max(1, smoke_test_n // 2)),
                selected_val[~selected_val.is_positive].head(max(1, smoke_test_n // 2)),
            ]
        ).reset_index(drop=True)

    print(
        f"[pilot data] train: {len(selected_train)} rollouts "
        f"({int(selected_train.is_positive.sum())} positive); "
        f"val: {len(selected_val)} rollouts ({int(selected_val.is_positive.sum())} positive)"
    )

    train_items = build_onset_probing_items(dataset_config, selected_train)
    val_items = build_onset_probing_items(dataset_config, selected_val)

    train_ds_config = TokenizedProbingDatasetConfig(
        dataset_id="onset_lora_pilot_train",
        hf_repo="local",
        split="train",
        max_length=max_length,
        max_completion_length=max_completion_length,
        shuffle=True,
        seed=seed,
    )
    # split="validation" (not "val") matches ProbeTrainer.evaluate()'s own check for
    # which dataset to also log as "val_loss" -- see trainer.py.
    val_ds_config = TokenizedProbingDatasetConfig(
        dataset_id="onset_lora_pilot_val",
        hf_repo="local",
        split="validation",
        max_length=max_length,
        max_completion_length=max_completion_length,
        shuffle=False,
        seed=seed,
    )

    weights = horizon_weights_for(selected_train, horizons)
    print(f"[pilot data] onset_horizon_weights (train, corpus-true inverse positive rate): {dict(zip(horizons, weights))}")

    return train_items, val_items, train_ds_config, val_ds_config, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=str, default=str(DEFAULT_DATASET_CONFIG_PATH))
    parser.add_argument("--layer", type=int, default=16, help="Layer to attach the probe to (default: 16, mid-network per subtask C's sweep).")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--probe-head-type", type=str, default="linear", choices=["linear", "attention"])
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument(
        "--smoke-test-n", type=int, default=0,
        help="If > 0, restrict to this many positive + this many negative rollouts (and half that "
        "for val) for a fast integration test instead of the real pilot sizing.",
    )
    args = parser.parse_args()

    load_dotenv()
    torch.manual_seed(args.seed)

    dataset_config = DatasetGenConfig.from_yaml(args.dataset_config)

    train_items, val_items, train_ds_config, val_ds_config, horizon_weights = build_pilot_datasets(
        dataset_config, args.horizons, args.max_length, args.max_completion_length, args.seed,
        smoke_test_n=args.smoke_test_n,
    )

    # Naming reflects what's actually being trained: len(horizons) > 1 is the true
    # multi-horizon design; len(horizons) == 1 is a single probe_N run through the
    # exact same "onset_multi_horizon" task machinery (compute_multi_horizon_bce_loss
    # etc. are already generic over any horizons list length, including 1 -- see
    # subtask H). Keeping one script for both avoids duplicating the training loop.
    if len(args.horizons) == 1:
        design_name = f"onset_probe_{args.horizons[0]}_lora"
    else:
        design_name = "onset_multi_horizon_lora"
    probe_id = f"{design_name}_pilot_layer{args.layer}" + ("_smoketest" if args.smoke_test_n else "")
    training_config = TrainingConfig(
        task="onset_multi_horizon",
        wandb_project="onset-probes",
        wandb_name=probe_id,
        probe_config=ProbeConfig(
            probe_id=probe_id,
            model_name=dataset_config.model_name,
            layer=args.layer,
            lora_layers="all",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            probe_dtype="float32",
            normalize_before_head="layernorm",
            probe_head_type=args.probe_head_type,
            n_outputs=len(args.horizons),
        ),
        onset_horizons=args.horizons,
        onset_horizon_weights=horizon_weights,
        model_dtype="bfloat16",
        probe_head_lr=1e-4,
        lora_lr=1e-4,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        enable_gradient_checkpointing=True,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        logging_steps=args.logging_steps,
        seed=args.seed,
        save_evaluation_metrics=True,
    )

    print("Training config:")
    for key, value in training_config.__dict__.items():
        if key not in ("train_dataset_configs", "eval_dataset_configs"):
            print(f"\t{key}: {value}")

    print(f"Loading model: {training_config.probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        training_config.probe_config.model_name,
        torch_dtype=resolve_torch_dtype(training_config.model_dtype),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if training_config.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print(f"Setting up probe: {probe_id}")
    model, probe = setup_probe(model, training_config.probe_config, seed=training_config.seed)
    print_trainable_parameters(probe)

    train_dataset = TokenizedProbingDataset(items=train_items, config=train_ds_config, tokenizer=tokenizer)
    val_dataset = TokenizedProbingDataset(items=val_items, config=val_ds_config, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(training_config.probe_config.probe_path),
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        num_train_epochs=training_config.num_train_epochs,
        logging_steps=training_config.logging_steps,
        eval_steps=training_config.eval_steps,
        remove_unused_columns=False,
        label_names=["classification_labels", "lm_labels"],
        report_to=args.report_to if args.report_to != "none" else [],
        run_name=probe_id,
        eval_strategy=training_config.evaluation_strategy,
        logging_first_step=True,
        logging_strategy="steps",
        max_grad_norm=training_config.max_grad_norm,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        seed=training_config.seed,
    )
    training_args.probe_head_lr = training_config.probe_head_lr
    training_args.lora_lr = training_config.lora_lr
    training_args.set_save(strategy="no")  # match train_probe.py -- save explicitly via callbacks/atexit instead

    if args.report_to == "wandb":
        import wandb
        wandb.init(project=training_config.wandb_project, name=probe_id, tags=training_config.wandb_tags or None)

    trainer = ProbeTrainer(
        probe=probe,
        eval_datasets=[val_dataset],
        cfg=training_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=tokenized_probing_collate_fn,
        eval_steps=training_config.eval_steps,
        tokenizer=tokenizer,
    )

    def save_checkpoint():
        probe_path = training_config.probe_config.probe_path
        probe_path.mkdir(parents=True, exist_ok=True)
        probe.save(probe_path)
        save_json(training_config, probe_path / "training_config.json")
        print(f"Saved checkpoint to {probe_path}")

    atexit.register(save_checkpoint)

    print("Training...")
    trainer.train()

    print(f"Saving model to {training_config.probe_config.probe_path}")
    save_checkpoint()

    trainer._last_eval_metrics = None
    eval_metrics = trainer.evaluate(verbose=True)
    save_json(eval_metrics, training_config.probe_config.probe_path / "evaluation_results.json")
    print("Final eval metrics:", eval_metrics)

    if args.report_to == "wandb":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
