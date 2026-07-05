"""Manifest read/write for the dataset-v2 pilot.

The manifest captures everything needed to reproduce/audit a dataset build:
the pipeline's git commit, a build timestamp, the full ``PilotDatasetConfig``
dump, and a handful of fields from the target model's ``config.json``
(hidden_size, num_hidden_layers, vocab_size) so downstream activation-shape
assumptions can be checked without re-loading the model.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from degeneration_probe.dataset_gen.config import PilotDatasetConfig
from degeneration_probe.dataset_gen.paths import manifest_path

# Default local HF cache to check before hitting the network. Overridable via
# the `hf_cache_dir` argument to `get_model_config` / `build_manifest`.
DEFAULT_HF_CACHE_DIR = Path("/capstor/scratch/cscs/mdenegri/degeneration-probe/hf_cache")

MODEL_CONFIG_FIELDS = ("hidden_size", "num_hidden_layers", "vocab_size", "model_type")


def get_git_commit(repo_root: Optional[Union[str, Path]] = None) -> str:
    """Return the current git commit hash of the pipeline, or "unknown" if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_model_config(
    model_name: str, hf_cache_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Fetch a few load-bearing fields from the target model's ``config.json``.

    Looks in the local HF cache first (no network needed if already cached),
    and only falls back to a live fetch from the HF Hub if the file isn't
    cached yet. Raises RuntimeError if neither works.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else str(DEFAULT_HF_CACHE_DIR)

    config_path = None
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        config_path = None

    if config_path is None:
        try:
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                cache_dir=cache_dir,
                local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve config.json for model {model_name!r} from the "
                f"local HF cache ({cache_dir}) or the network. Original error: {exc}"
            ) from exc

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    return {field_name: raw_config.get(field_name) for field_name in MODEL_CONFIG_FIELDS}


def build_manifest(
    config: PilotDatasetConfig,
    repo_root: Optional[Union[str, Path]] = None,
    hf_cache_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Assemble the manifest dict (does not write it to disk)."""
    return {
        "pipeline_git_commit": get_git_commit(repo_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "model_config": get_model_config(config.model_name, hf_cache_dir=hf_cache_dir),
    }


def write_manifest(
    config: PilotDatasetConfig,
    path: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    hf_cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Build the manifest and write it to ``path`` (default: paths.manifest_path(config))."""
    out_path = Path(path) if path is not None else manifest_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(config, repo_root=repo_root, hf_cache_dir=hf_cache_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    return out_path


def read_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
