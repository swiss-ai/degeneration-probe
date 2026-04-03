"""YAML config loading and experiment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class DatasetConfig:
    name: str
    subset: Optional[str] = None
    split: str = "train"
    prompt_field: Optional[str] = None
    max_samples: int = 25
    shuffle: bool = False
    seed: int = 42
    max_prompts: int = 10


@dataclass
class AnalysisConfig:
    chunk_size: int = 256
    n_values: List[int] = field(default_factory=lambda: [1, 3])
    save_hidden_states: bool = False


@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    analysis: AnalysisConfig
    output_dir: Optional[Path] = None  # set by run.py at runtime, not from YAML


def load_config(path: Path) -> Config:
    """Load a YAML experiment config file and return a Config instance."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    for key in ("model", "dataset", "analysis"):
        if key not in raw:
            raise ValueError(f"Config file '{path}' is missing required top-level key: '{key}'")

    return Config(
        model=ModelConfig(**raw["model"]),
        dataset=DatasetConfig(**raw["dataset"]),
        analysis=AnalysisConfig(**raw["analysis"]),
    )


# ---------------------------------------------------------------------------
# Phase 2 — Probe training configuration (used by scripts/train.py via Hydra)
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    """Configuration for the probe architecture."""

    layer: int = -1  # -1 = last layer
    pooling: str = "mean"  # mean | max | last_token
    threshold: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for the probe training loop."""

    # Probe
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    # Data — list of JSONL paths; model is auto-resolved from data
    train_data: List[str] = field(default_factory=list)
    eval_data: Optional[str] = None  # if None, split from train
    eval_fraction: float = 0.2
    max_length: int = 2048

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 4
    num_epochs: int = 10
    seed: int = 42
    pos_weight: Optional[float] = None  # weight for positive class in BCE

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Output
    output_dir: str = "outputs/probes"


def load_training_config(path: Path) -> TrainingConfig:
    """Load a plain YAML training config and return a TrainingConfig instance."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Normalize train_data: accept a single string or a list
    train_data = raw.get("train_data", [])
    if isinstance(train_data, str):
        train_data = [train_data]

    probe_raw = raw.get("probe", {})
    probe = ProbeConfig(**probe_raw)

    return TrainingConfig(
        probe=probe,
        train_data=train_data,
        eval_data=raw.get("eval_data"),
        eval_fraction=raw.get("eval_fraction", 0.2),
        max_length=raw.get("max_length", 2048),
        learning_rate=raw.get("learning_rate", 1e-3),
        batch_size=raw.get("batch_size", 4),
        num_epochs=raw.get("num_epochs", 10),
        seed=raw.get("seed", 42),
        pos_weight=raw.get("pos_weight"),
        wandb_project=raw.get("wandb_project"),
        wandb_run_name=raw.get("wandb_run_name"),
        output_dir=raw.get("output_dir", "outputs/probes"),
    )
