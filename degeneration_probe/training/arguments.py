"""Translation from the experiment configuration to trainer arguments.

Kept apart from the training entry point so the combination of cadences,
checkpoint rotation and best-model selection can be checked without allocating
a language model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from transformers import TrainingArguments

from degeneration_probe.config import ExperimentConfig


def resolve_token_budget(
    tokens_per_step: int,
    *,
    valid_tokens: int,
    examples: int,
    per_device_batch_size: int,
) -> Dict[str, float]:
    """Size the accumulation so each step sees the requested tokens.

    Measured from the training stream rather than assumed from the window size.
    A rule that emits windows and a rule that keeps whole rollouts leave very
    different numbers of tokens behind per example, so a budget derived from the
    configuration would give one rule several times the tokens of another, and
    that difference would sit inside a comparison built to isolate the rule.
    """
    tokens_per_example = valid_tokens / max(1, examples)
    per_batch = per_device_batch_size * tokens_per_example
    accumulation = max(1, round(tokens_per_step / per_batch)) if per_batch > 0 else 1
    return {
        "tokens_per_step_requested": int(tokens_per_step),
        "tokens_per_step_realized": round(accumulation * per_batch),
        "tokens_per_example": tokens_per_example,
        "gradient_accumulation_steps": accumulation,
    }


def build_training_arguments(
    config: ExperimentConfig,
    *,
    run_dir: Path,
    run_name: str,
    report_to: List[str],
    use_cpu: bool = False,
) -> TrainingArguments:
    runtime = config.training.runtime
    validation = config.training.validation
    checkpoint = config.training.checkpoint
    # Clipping a norm taken across every head at once would couple heads that
    # otherwise share nothing, so a multi-head probe turns the global clip off
    # and the trainer reapplies the same limit to each head separately.
    max_grad_norm = (
        0.0 if config.training.probe.layers is not None else runtime.max_grad_norm
    )
    return TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=runtime.per_device_train_batch_size,
        per_device_eval_batch_size=runtime.per_device_eval_batch_size,
        gradient_accumulation_steps=runtime.gradient_accumulation_steps,
        num_train_epochs=runtime.num_train_epochs,
        max_steps=runtime.max_steps,
        max_grad_norm=max_grad_norm,
        logging_steps=runtime.logging_steps,
        dataloader_num_workers=runtime.dataloader_num_workers,
        eval_strategy=validation.strategy,
        eval_steps=validation.steps,
        save_strategy=checkpoint.strategy,
        save_steps=checkpoint.steps or 500,
        save_total_limit=checkpoint.save_total_limit,
        load_best_model_at_end=checkpoint.keep_best,
        metric_for_best_model=checkpoint.metric_for_best_model,
        greater_is_better=checkpoint.greater_is_better,
        remove_unused_columns=False,
        label_names=["targets"],
        report_to=report_to,
        run_name=run_name,
        learning_rate=config.training.optimizer.probe_learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        seed=runtime.seed,
        logging_first_step=True,
        use_cpu=use_cpu,
    )
