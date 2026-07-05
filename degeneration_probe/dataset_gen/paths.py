"""Path layout for the dataset-v2 pilot.

Every path the dataset-gen pipeline reads or writes is computed here, given a
``PilotDatasetConfig`` and, where applicable, a ``domain`` ("in_domain" or
"held_out"), a ``prompt_id``, and/or a rollout index. Centralizing this avoids
path-format drift between the (separately implemented) generation, labeling,
and activation-extraction stages.

Layout under ``config.output_root``::

    dataset_v2_pilot/
        manifest.json
        prompts/prompts.parquet
        generations/<domain>/shard_00000.parquet
        labels/<domain>/shard_00000.parquet
        prompt_stats/prompt_stats.parquet
        activations/<domain>/<prompt_id>/rollout_<k>.safetensors
        activations/manifest.parquet
        llm_judge/
        splits/<split_name>.jsonl
        notebooks/
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from degeneration_probe.dataset_gen.config import PilotDatasetConfig

Domain = Literal["in_domain", "held_out"]

VALID_DOMAINS = ("in_domain", "held_out")


def _check_domain(domain: str) -> None:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"domain must be one of {VALID_DOMAINS}, got {domain!r}")


# --- top level -------------------------------------------------------------

def dataset_root(config: PilotDatasetConfig) -> Path:
    return Path(config.output_root)


def manifest_path(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "manifest.json"


# --- prompts -----------------------------------------------------------------

def prompts_dir(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "prompts"


def prompts_path(config: PilotDatasetConfig) -> Path:
    return prompts_dir(config) / "prompts.parquet"


# --- generations ---------------------------------------------------------------

def generations_dir(config: PilotDatasetConfig, domain: Domain) -> Path:
    _check_domain(domain)
    return dataset_root(config) / "generations" / domain


def generations_shard_path(config: PilotDatasetConfig, domain: Domain, shard_idx: int) -> Path:
    return generations_dir(config, domain) / f"shard_{shard_idx:05d}.parquet"


# --- labels --------------------------------------------------------------------

def labels_dir(config: PilotDatasetConfig, domain: Domain) -> Path:
    _check_domain(domain)
    return dataset_root(config) / "labels" / domain


def labels_shard_path(config: PilotDatasetConfig, domain: Domain, shard_idx: int) -> Path:
    return labels_dir(config, domain) / f"shard_{shard_idx:05d}.parquet"


# --- prompt stats ----------------------------------------------------------------

def prompt_stats_dir(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "prompt_stats"


def prompt_stats_path(config: PilotDatasetConfig) -> Path:
    return prompt_stats_dir(config) / "prompt_stats.parquet"


# --- activations -----------------------------------------------------------------

def activations_root(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "activations"


def activations_domain_dir(config: PilotDatasetConfig, domain: Domain) -> Path:
    _check_domain(domain)
    return activations_root(config) / domain


def activations_prompt_dir(config: PilotDatasetConfig, domain: Domain, prompt_id: str) -> Path:
    return activations_domain_dir(config, domain) / str(prompt_id)


def rollout_activation_path(
    config: PilotDatasetConfig, domain: Domain, prompt_id: str, rollout_idx: int
) -> Path:
    return activations_prompt_dir(config, domain, prompt_id) / f"rollout_{rollout_idx}.safetensors"


def activations_manifest_path(config: PilotDatasetConfig) -> Path:
    return activations_root(config) / "manifest.parquet"


# --- llm judge -----------------------------------------------------------------

def llm_judge_dir(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "llm_judge"


# --- splits ----------------------------------------------------------------------

def splits_dir(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "splits"


def split_path(config: PilotDatasetConfig, split_name: str) -> Path:
    return splits_dir(config) / f"{split_name}.jsonl"


# --- notebooks -------------------------------------------------------------------

def notebooks_dir(config: PilotDatasetConfig) -> Path:
    return dataset_root(config) / "notebooks"


# --- work root (scratch, resumable intermediates) ---------------------------------

def work_root(config: PilotDatasetConfig) -> Path:
    return Path(config.work_root)


def all_output_dirs(config: PilotDatasetConfig) -> list[Path]:
    """All directories that should exist (possibly empty) under output_root."""
    dirs = [
        dataset_root(config),
        prompts_dir(config),
        prompt_stats_dir(config),
        activations_root(config),
        llm_judge_dir(config),
        splits_dir(config),
        notebooks_dir(config),
    ]
    for domain in VALID_DOMAINS:
        dirs.append(generations_dir(config, domain))
        dirs.append(labels_dir(config, domain))
        dirs.append(activations_domain_dir(config, domain))
    return dirs
