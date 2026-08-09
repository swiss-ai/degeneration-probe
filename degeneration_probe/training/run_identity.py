"""How a training run names, groups and labels itself.

A sweep is only readable if every run says, in its own name and tags, which
axes it varies. Names and groups are therefore derived from the configuration
rather than written by hand:

- the **name** carries the axes a human scans for, plus a fingerprint of the
  full configuration, so two runs that differ in any setting can never share a
  name (and therefore never share an output directory),
- the **group** is the name with the seed removed, so seed repeats of one
  recipe aggregate into a single line with a spread,
- the **tags** are ``key:value`` labels, one per axis, which is what filtering
  and grouping in the tracking UI actually keys off.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List

from degeneration_probe.config import ExperimentConfig

# Settings that describe how a run is executed or observed, never what it
# learns. They are excluded from the fingerprint so that re-running a
# configuration with different logging or checkpointing lands on the same
# identity, and can therefore resume.
VOLATILE_TRAINING_FIELDS = {"checkpoint", "wandb", "validation"}
VOLATILE_RUNTIME_FIELDS = {
    "logging_steps",
    "dataloader_num_workers",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
}


def lora_scope(config: ExperimentConfig) -> str:
    """A short, stable label for how much of the model adapts."""
    lora = config.training.lora
    if not lora.enabled or lora.layers == "none":
        return "none"
    if lora.layers == "all":
        return "all"
    return f"custom{len(lora.layers)}"


def run_axes(config: ExperimentConfig) -> Dict[str, Any]:
    """Everything that distinguishes this run from another one.

    Recorded with the run, sent to the tracker as configuration, and turned
    into the tags below, so a later comparison never has to parse a run name.
    """
    training = config.training
    scope = lora_scope(config)
    return {
        "model": config.model.short_name,
        "dataset": config.dataset.short_name,
        "task": training.task,
        "layer": training.probe.layer,
        "context_window": training.probe.context_window_size,
        "normalization": training.probe.normalization,
        "label": training.label.family,
        "horizon": training.label.horizon if training.label.family == "frontier_hard" else None,
        "decay": f"{training.label.decay}{training.label.decay_length:g}"
        if training.label.family == "frontier_soft"
        else None,
        "signal": training.label.signal if training.label.family == "token_signal" else None,
        "loss": training.loss.name,
        "pos_weight": "on"
        if training.loss.name == "bce" and training.loss.bce.use_pos_weight
        else "off",
        "lora": scope,
        "lora_rank": training.lora.rank if scope != "none" else 0,
        "negatives_per_positive": config.dataset.sampling.train_negative_rollouts_per_positive,
        "probe_lr": training.optimizer.probe_learning_rate,
        "lora_lr": training.optimizer.lora_learning_rate if scope != "none" else None,
        "epochs": training.runtime.num_train_epochs,
        "max_steps": training.runtime.max_steps,
        "seed": training.runtime.seed,
    }


def _fingerprint_payload(config: ExperimentConfig, *, ignore_seed: bool) -> Dict[str, Any]:
    payload = asdict(config)
    training = payload["training"]
    for field_name in VOLATILE_TRAINING_FIELDS:
        training.pop(field_name, None)
    for field_name in VOLATILE_RUNTIME_FIELDS:
        training["runtime"].pop(field_name, None)
    if ignore_seed:
        training["runtime"].pop("seed", None)
        payload["dataset"]["sampling"].pop("seed", None)
    return payload


def _fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def config_fingerprint(config: ExperimentConfig) -> str:
    """Short hash of every setting that changes what the run learns."""
    return _fingerprint(_fingerprint_payload(config, ignore_seed=False))


def recipe_fingerprint(config: ExperimentConfig) -> str:
    """The same hash with seeds removed, shared by every repeat of a recipe."""
    return _fingerprint(_fingerprint_payload(config, ignore_seed=True))


def _readable_prefix(config: ExperimentConfig) -> str:
    training = config.training
    label = training.label.family.replace("frontier_", "")
    if training.label.family == "frontier_hard" and training.label.horizon:
        label = f"{label}{training.label.horizon}"
    elif training.label.family == "token_signal":
        label = training.label.signal
    return (
        f"{config.dataset.short_name}"
        f"_L{training.probe.layer}"
        f"_{label}"
        f"_{training.loss.name}"
        f"_lora-{lora_scope(config)}"
    )


def derive_run_name(config: ExperimentConfig) -> str:
    """Readable axes, the seed, and a fingerprint that guarantees uniqueness."""
    if config.training.wandb.name:
        return config.training.wandb.name
    return (
        f"{_readable_prefix(config)}"
        f"_s{config.training.runtime.seed}"
        f"_{config_fingerprint(config)}"
    )


def derive_group_name(config: ExperimentConfig) -> str:
    """The run name without its seed: one line per recipe, spread over seeds."""
    if config.training.wandb.group:
        return config.training.wandb.group
    return f"{_readable_prefix(config)}_{recipe_fingerprint(config)}"


def derive_tags(config: ExperimentConfig) -> List[str]:
    """One ``key:value`` tag per axis, plus whatever the config adds."""
    tags = [
        f"{key}:{value}"
        for key, value in run_axes(config).items()
        if value is not None
    ]
    tags.append(f"recipe:{recipe_fingerprint(config)}")
    tags.append(f"config:{config_fingerprint(config)}")
    tags.extend(config.training.wandb.tags)
    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique
