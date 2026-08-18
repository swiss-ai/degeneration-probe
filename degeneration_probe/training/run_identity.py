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
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Dict, List

from degeneration_probe.config import ExperimentConfig

# Settings that describe how a run is executed or observed, never what it
# learns. They are excluded from the fingerprint so that re-running a
# configuration with different logging or checkpointing lands on the same
# identity, and can therefore resume.
# ``stopping`` sits here because every checkpoint of a run is kept: two runs
# differing only in when they stop are one run truncated at two points, and the
# shorter one's trajectory is a prefix of the longer one's. Sharing an identity
# is what lets a rule with more patience continue a run rather than repeat it,
# and what stops a re-reading of the same saved checkpoints from looking like a
# different experiment.
VOLATILE_TRAINING_FIELDS = {"checkpoint", "wandb", "validation", "stopping"}
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
        # A run reading one depth has a layer; a run reading many has a span
        # and no layer, so that nothing downstream mistakes the leftover
        # single-layer setting for the depth a result came from.
        "layer": training.probe.layer if training.probe.layers is None else None,
        "layers": depth_label(training.probe)[1:] if training.probe.layers is not None else None,
        "context_window": training.probe.context_window_size,
        "normalization": training.probe.normalization,
        "features": training.features.regime,
        "selection": training.selection.strategy,
        "window": training.selection.window_size
        if training.selection.strategy != "all_tokens"
        else None,
        "anchor": training.selection.anchor
        if training.selection.strategy.startswith("frontier_window")
        else None,
        "tokens_per_step": training.budget.tokens_per_step,
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


# A field with no default states something the configuration cannot do without,
# so it is carried whatever its value.
_REQUIRED = object()


def _declared_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return _REQUIRED


def _chosen(instance: Any) -> Dict[str, Any]:
    """Only the settings that differ from what the configuration already says.

    A configuration gains fields over time, and a field nobody set describes no
    decision. Hashing the whole structure meant that adding one renamed every
    run that already existed, which quietly detached each of them from the runs
    they were meant to continue: a resumed run derives its own name, finds no
    directory under it, and starts again from nothing. Comparing against the
    declared defaults keeps an addition invisible until somebody uses it.
    """
    payload: Dict[str, Any] = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if is_dataclass(value):
            nested = _chosen(value)
            if nested:
                payload[field.name] = nested
            continue
        default = _declared_default(field)
        if default is not _REQUIRED and value == default:
            continue
        payload[field.name] = value
    return payload


def _prune(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop sections that removing a setting left empty.

    A section holding nothing says the same as no section at all, and the two
    must hash alike or the volatile settings would be excluded in name only.
    """
    pruned = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            value = _prune(value)
            if not value:
                continue
        pruned[key] = value
    return pruned


def _fingerprint_payload(config: ExperimentConfig, *, ignore_seed: bool) -> Dict[str, Any]:
    payload = _chosen(config)
    training = payload.get("training", {})
    for field_name in VOLATILE_TRAINING_FIELDS:
        training.pop(field_name, None)
    runtime = training.get("runtime", {})
    for field_name in VOLATILE_RUNTIME_FIELDS:
        runtime.pop(field_name, None)
    if ignore_seed:
        runtime.pop("seed", None)
        payload.get("dataset", {}).get("sampling", {}).pop("seed", None)
    return _prune(payload)


def _fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def config_fingerprint(config: ExperimentConfig) -> str:
    """Short hash of every setting that changes what the run learns."""
    return _fingerprint(_fingerprint_payload(config, ignore_seed=False))


def recipe_fingerprint(config: ExperimentConfig) -> str:
    """The same hash with seeds removed, shared by every repeat of a recipe."""
    return _fingerprint(_fingerprint_payload(config, ignore_seed=True))


def depth_label(probe) -> str:
    """How a run names the depths it reads.

    A run that carries a head at every depth belongs to no single layer, so
    naming it after one would be a claim about the run that is not true.
    """
    layers = sorted(probe.probed_layers)
    if len(layers) == 1:
        return f"L{layers[0]}"
    if layers == list(range(layers[0], layers[-1] + 1)):
        return f"L{layers[0]}-{layers[-1]}"
    return f"L{len(layers)}x"


def _readable_prefix(config: ExperimentConfig) -> str:
    training = config.training
    label = training.label.family.replace("frontier_", "")
    if training.label.family == "frontier_hard" and training.label.horizon:
        label = f"{label}{training.label.horizon}"
    elif training.label.family == "token_signal":
        label = training.label.signal
    selection = training.selection.strategy.replace("frontier_window", "frontier")
    if training.selection.strategy != "all_tokens":
        selection = f"{selection}{training.selection.window_size}"
    return (
        f"{config.dataset.short_name}"
        f"_{depth_label(training.probe)}"
        f"_{selection}"
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
