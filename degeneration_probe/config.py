"""Typed configuration for the single ``degeneration`` training task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _as_float(value: Any) -> float:
    return float(value) if isinstance(value, str) else value


@dataclass
class ModelConfig:
    short_name: str
    name: str
    tokenizer_name: Optional[str] = None
    dtype: str = "bfloat16"


@dataclass
class ProbeConfig:
    id: str
    layer: int = 30
    dtype: str = "float32"
    normalization: str = "layernorm"
    context_window_size: int = 1

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("training.probe.layer must be non-negative")
        if self.context_window_size < 1:
            raise ValueError("training.probe.context_window_size must be at least 1")
        if self.normalization not in {"none", "layernorm", "rmsnorm", "l2"}:
            raise ValueError("training.probe.normalization must be none, layernorm, rmsnorm, or l2")


@dataclass
class LoraConfig:
    enabled: bool = True
    layers: Union[str, List[int]] = "all"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05

    def __post_init__(self) -> None:
        self.dropout = _as_float(self.dropout)
        if isinstance(self.layers, str) and self.layers not in {"all", "none"}:
            raise ValueError("training.lora.layers must be 'all', 'none', or a list of layer indices")


@dataclass
class BceLossConfig:
    use_pos_weight: bool = True


@dataclass
class MseLossConfig:
    output_activation: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.output_activation != "sigmoid":
            raise ValueError("Only sigmoid is supported for training.loss.mse.output_activation")


@dataclass
class LossConfig:
    name: str = "bce"
    bce: BceLossConfig = field(default_factory=BceLossConfig)
    mse: MseLossConfig = field(default_factory=MseLossConfig)

    def __post_init__(self) -> None:
        if isinstance(self.bce, dict):
            self.bce = BceLossConfig(**self.bce)
        if isinstance(self.mse, dict):
            self.mse = MseLossConfig(**self.mse)
        if self.name not in {"bce", "mse"}:
            raise ValueError("training.loss.name must be either 'bce' or 'mse'")


@dataclass
class OptimizerConfig:
    probe_learning_rate: float = 1e-4
    lora_learning_rate: float = 1e-4
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        self.probe_learning_rate = _as_float(self.probe_learning_rate)
        self.lora_learning_rate = _as_float(self.lora_learning_rate)
        self.weight_decay = _as_float(self.weight_decay)


@dataclass
class RuntimeConfig:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    max_steps: int = -1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    dataloader_num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        self.num_train_epochs = _as_float(self.num_train_epochs)
        self.max_grad_norm = _as_float(self.max_grad_norm)


@dataclass
class ValidationConfig:
    strategy: str = "epoch"
    steps: Optional[int] = None
    metrics: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.strategy not in {"no", "steps", "epoch"}:
            raise ValueError("training.validation.strategy must be no, steps, or epoch")
        if self.strategy == "steps" and not self.steps:
            raise ValueError("training.validation.steps is required when strategy='steps'")


@dataclass
class CheckpointConfig:
    output_dir: str = "outputs/degeneration_probe"
    save_each_epoch: bool = True

    @property
    def path(self) -> Path:
        return Path(self.output_dir)


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "degeneration-probes"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("training.wandb.mode must be online, offline, or disabled")


@dataclass
class TrainingConfig:
    task: str
    short_name: str
    probe: ProbeConfig
    lora: LoraConfig = field(default_factory=LoraConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        nested = {
            "probe": ProbeConfig,
            "lora": LoraConfig,
            "loss": LossConfig,
            "optimizer": OptimizerConfig,
            "runtime": RuntimeConfig,
            "validation": ValidationConfig,
            "checkpoint": CheckpointConfig,
            "wandb": WandbConfig,
        }
        for name, cls in nested.items():
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))
        if self.task != "degeneration":
            raise ValueError("The only supported task is 'degeneration'")


@dataclass
class SplitConfig:
    train: str = "train"
    validation: str = "val"
    test_indomain: str = "test_indomain"
    test_heldout_domains: str = "test_heldout_domains"

    @property
    def final_evaluation(self) -> List[str]:
        return [self.validation, self.test_indomain, self.test_heldout_domains]


@dataclass
class TokenizationConfig:
    max_length: int = 4608
    max_completion_length: int = 4096
    prompt_truncation_side: str = "left"

    def __post_init__(self) -> None:
        if self.max_length < 1 or self.max_completion_length < 1:
            raise ValueError("dataset.tokenization lengths must be positive")
        if self.max_completion_length > self.max_length:
            raise ValueError("dataset.tokenization.max_completion_length cannot exceed max_length")
        if self.prompt_truncation_side not in {"left", "right"}:
            raise ValueError("dataset.tokenization.prompt_truncation_side must be left or right")


@dataclass
class SamplingConfig:
    train_negative_rollouts_per_positive: Optional[float] = 4.0
    evaluation_negative_rollouts_per_positive: Optional[float] = None
    domain_stratified: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("train_negative_rollouts_per_positive", "evaluation_negative_rollouts_per_positive"):
            value = getattr(self, name)
            if value is not None:
                value = _as_float(value)
                if value < 0:
                    raise ValueError(f"dataset.sampling.{name} must be non-negative or null")
                setattr(self, name, value)


@dataclass
class DatasetConfig:
    short_name: str
    build_root: str
    build_config: str
    splits: SplitConfig = field(default_factory=SplitConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        for name, cls in {
            "splits": SplitConfig,
            "tokenization": TokenizationConfig,
            "sampling": SamplingConfig,
        }.items():
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))


@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

    def __post_init__(self) -> None:
        for name, cls in {
            "model": ModelConfig,
            "training": TrainingConfig,
            "dataset": DatasetConfig,
        }.items():
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ExperimentConfig":
        unexpected = set(config) - {"model", "training", "dataset"}
        if unexpected:
            raise ValueError(f"Unexpected top-level config sections: {sorted(unexpected)}")
        return cls(**config)
