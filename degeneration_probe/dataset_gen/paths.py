"""Path layout for the dataset generation pipeline.

Every path the dataset-gen pipeline reads or writes is computed here, given a
``DatasetGenConfig`` and, where applicable, a ``domain``, a ``prompt_id``,
and/or a rollout index. Centralizing this avoids path-format drift between
the (separately implemented) generation, labeling, and activation-extraction
stages.

IMPORTANT: ``domain`` here means one specific *source* name (e.g.
"deepmath_103k", "numinamath_1_5", "aime_2025", ...) -- i.e. one entry from
``config.in_domain_sources + config.held_out_sources`` -- not the coarser
"in_domain" vs. "held_out" split. Each source gets its own subfolder so
later phases (per-source degeneration-rate stats, LLM-judge calibration
that needs to isolate e.g. the math sources) can see a clean per-source
breakdown. Use ``is_held_out_domain`` to check which split a given source
belongs to without collapsing the folder structure.

Layout under ``config.output_root`` (e.g.
``degeneration-dataset-apertus-8b-instruct/``)::

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
from typing import Set

from degeneration_probe.dataset_gen.config import DatasetGenConfig

# A "domain" is a source name, e.g. "deepmath_103k" or "aime_2025" -- see the
# module docstring. There is no fixed set of values; it's whatever is
# configured in `config.in_domain_sources` / `config.held_out_sources`.
Domain = str


def _in_domain_names(config: DatasetGenConfig) -> Set[str]:
    return {source["name"] for source in config.in_domain_sources}


def _held_out_domain_names(config: DatasetGenConfig) -> Set[str]:
    return {source["name"] for source in config.held_out_sources}


def configured_domain_names(config: DatasetGenConfig) -> Set[str]:
    """All valid domain (source) names configured for this dataset build."""
    return _in_domain_names(config) | _held_out_domain_names(config)


def is_held_out_domain(config: DatasetGenConfig, domain: str) -> bool:
    """Whether `domain` (a source name) belongs to config.held_out_sources.

    Later phases (e.g. split assignment) need to know this to route a given
    source's prompts to the right split without merging its folder with
    others.
    """
    _check_domain(config, domain)
    return domain in _held_out_domain_names(config)


def _check_domain(config: DatasetGenConfig, domain: str) -> None:
    valid = configured_domain_names(config)
    if domain not in valid:
        raise ValueError(
            f"domain must be one of the configured source names {sorted(valid)}, "
            f"got {domain!r}"
        )


# --- top level -------------------------------------------------------------

def dataset_root(config: DatasetGenConfig) -> Path:
    return Path(config.output_root)


def manifest_path(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "manifest.json"


# --- prompts -----------------------------------------------------------------

def prompts_dir(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "prompts"


def prompts_path(config: DatasetGenConfig) -> Path:
    return prompts_dir(config) / "prompts.parquet"


# --- generations ---------------------------------------------------------------

def generations_dir(config: DatasetGenConfig, domain: Domain) -> Path:
    _check_domain(config, domain)
    return dataset_root(config) / "generations" / domain


def generations_shard_path(config: DatasetGenConfig, domain: Domain, shard_idx: int) -> Path:
    return generations_dir(config, domain) / f"shard_{shard_idx:05d}.parquet"


# --- labels --------------------------------------------------------------------

def labels_dir(config: DatasetGenConfig, domain: Domain) -> Path:
    _check_domain(config, domain)
    return dataset_root(config) / "labels" / domain


def labels_shard_path(config: DatasetGenConfig, domain: Domain, shard_idx: int) -> Path:
    return labels_dir(config, domain) / f"shard_{shard_idx:05d}.parquet"


# --- prompt stats ----------------------------------------------------------------

def prompt_stats_dir(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "prompt_stats"


def prompt_stats_path(config: DatasetGenConfig) -> Path:
    return prompt_stats_dir(config) / "prompt_stats.parquet"


# --- activations -----------------------------------------------------------------

def activations_root(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "activations"


def activations_domain_dir(config: DatasetGenConfig, domain: Domain) -> Path:
    _check_domain(config, domain)
    return activations_root(config) / domain


def activations_prompt_dir(config: DatasetGenConfig, domain: Domain, prompt_id: str) -> Path:
    return activations_domain_dir(config, domain) / str(prompt_id)


def rollout_activation_path(
    config: DatasetGenConfig, domain: Domain, prompt_id: str, rollout_idx: int
) -> Path:
    return activations_prompt_dir(config, domain, prompt_id) / f"rollout_{rollout_idx}.safetensors"


def activations_manifest_path(config: DatasetGenConfig) -> Path:
    return activations_root(config) / "manifest.parquet"


# --- llm judge -----------------------------------------------------------------

def llm_judge_dir(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "llm_judge"


def llm_judge_sample_path(config: DatasetGenConfig) -> Path:
    """The selected calibration rollouts (prompt_id, rollout_idx, stratum, ...)
    to send to the judge -- built once by ``llm_judge.select_calibration_sample``
    so the sampling logic is reproducible rather than re-derived ad hoc."""
    return llm_judge_dir(config) / "calibration_sample.parquet"


def llm_judge_results_path(config: DatasetGenConfig, backend: str) -> Path:
    """Judge verdicts for one backend (e.g. "anthropic", "openrouter"). Kept
    per-backend so results from different judges never collide, mirroring
    how the pipeline's other resumable manifests (e.g.
    ``activations_manifest_path``) are keyed to avoid cross-run interference."""
    return llm_judge_dir(config) / f"results_{backend}.parquet"


# --- splits ----------------------------------------------------------------------

def splits_dir(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "splits"


def split_path(config: DatasetGenConfig, split_name: str) -> Path:
    return splits_dir(config) / f"{split_name}.jsonl"


# --- notebooks -------------------------------------------------------------------

def notebooks_dir(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "notebooks"


# --- onset labels (derived artifact for probe training, not part of the core
# generate/label/cache-activations pipeline -- see
# degeneration_probe/dataset_gen/onset_labels.py) --------------------------------

def onset_labels_dir(config: DatasetGenConfig) -> Path:
    return dataset_root(config) / "onset_labels"


def onset_labels_path(config: DatasetGenConfig) -> Path:
    return onset_labels_dir(config) / "onset_labels.parquet"


def onset_quote_positions_path(config: DatasetGenConfig) -> Path:
    """Judge quotes located in the token stream, one row per capped rollout.

    Cached separately from the labels because locating a quote is the expensive
    step and only changes when the judge is re-run.
    """
    return onset_labels_dir(config) / "onset_quote_positions.parquet"


# --- work root (scratch, resumable intermediates) ---------------------------------

def work_root(config: DatasetGenConfig) -> Path:
    return Path(config.work_root)


def all_output_dirs(config: DatasetGenConfig) -> list[Path]:
    """All directories that should exist (possibly empty) under output_root.

    One subfolder per configured source name under generations/, labels/,
    and activations/ (not per in_domain/held_out split).
    """
    dirs = [
        dataset_root(config),
        prompts_dir(config),
        prompt_stats_dir(config),
        activations_root(config),
        llm_judge_dir(config),
        splits_dir(config),
        notebooks_dir(config),
    ]
    for domain in sorted(configured_domain_names(config)):
        dirs.append(generations_dir(config, domain))
        dirs.append(labels_dir(config, domain))
        dirs.append(activations_domain_dir(config, domain))
    return dirs
