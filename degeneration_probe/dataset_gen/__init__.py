"""Dataset generation pipeline (scaffolding).

This package only contains the config schema, path layout, and manifest
read/write helpers for now. Generation, labeling, and activation-extraction
logic live elsewhere and are added separately; this package does not touch
``degeneration_probe/data`` (the training-time consumer of pre-built HF
datasets).
"""

from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.manifest import (
    build_manifest,
    get_git_commit,
    get_model_config,
    read_manifest,
    write_manifest,
)

__all__ = [
    "DatasetGenConfig",
    "build_manifest",
    "get_git_commit",
    "get_model_config",
    "read_manifest",
    "write_manifest",
]
