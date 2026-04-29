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
# Probe training configuration (degeneration regression probe)
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # Optional explicit layer list; default = layers 0..probe.layer (inclusive).
    layers: Optional[List[int]] = None


@dataclass
class ProbeConfig:
    """Probe architecture / hook configuration."""

    layer: int = 12  # middle-ish layer; override per model
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class LabelConfig:
    """How per-token 1 - TTR labels are computed from the completion."""

    window_size: int = 256  # TTR computed over next N tokens
    primary_n: int = 1  # n-gram size for TTR
    # If set, labels become binary: 1 iff TTR(next window) <= ttr_threshold,
    # and the trainer switches from MSE on sigmoid to BCE-with-logits.
    ttr_threshold: Optional[float] = None


@dataclass
class LearningRateConfig:
    """Per-component learning rates."""

    head: float = 5.0e-3
    lora: float = 5.0e-5


@dataclass
class HFDatasetConfig:
    """Optional HuggingFace dataset spec (alternative to local JSONL paths)."""

    name: str
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    max_train_rows: Optional[int] = None
    max_eval_rows: Optional[int] = None


@dataclass
class ModelOverride:
    """Explicit model name (required when training from HF data; auto-resolved
    from JSONL records otherwise)."""

    name: str


@dataclass
class TrainingConfig:
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    model: Optional[ModelOverride] = None

    train_data: List[str] = field(default_factory=list)
    eval_data: Optional[str] = None
    hf_dataset: Optional[HFDatasetConfig] = None
    eval_fraction: float = 0.2
    max_length: int = 2048

    label: LabelConfig = field(default_factory=LabelConfig)
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    batch_size: int = 4
    num_epochs: int = 10
    seed: int = 42

    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    output_dir: str = "outputs/probes"


def _probe_from_raw(raw: dict) -> ProbeConfig:
    lora_raw = raw.get("lora", {}) or {}
    lora = LoRAConfig(**lora_raw)
    kwargs = {k: v for k, v in raw.items() if k != "lora"}
    return ProbeConfig(lora=lora, **kwargs)


def load_training_config(path: Path) -> TrainingConfig:
    """Load a plain YAML training config and return a TrainingConfig instance."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    train_data = raw.get("train_data", [])
    if isinstance(train_data, str):
        train_data = [train_data]

    probe = _probe_from_raw(raw.get("probe", {}) or {})
    label = LabelConfig(**(raw.get("label", {}) or {}))
    learning_rate = LearningRateConfig(**(raw.get("learning_rate", {}) or {}))

    model = ModelOverride(**raw["model"]) if raw.get("model") else None
    hf_dataset = HFDatasetConfig(**raw["hf_dataset"]) if raw.get("hf_dataset") else None

    return TrainingConfig(
        probe=probe,
        model=model,
        train_data=train_data,
        eval_data=raw.get("eval_data"),
        hf_dataset=hf_dataset,
        eval_fraction=raw.get("eval_fraction", 0.2),
        max_length=raw.get("max_length", 2048),
        label=label,
        learning_rate=learning_rate,
        batch_size=raw.get("batch_size", 4),
        num_epochs=raw.get("num_epochs", 10),
        seed=raw.get("seed", 42),
        wandb_project=raw.get("wandb_project"),
        wandb_entity=raw.get("wandb_entity"),
        wandb_run_name=raw.get("wandb_run_name"),
        output_dir=raw.get("output_dir", "outputs/probes"),
    )
