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
class LabelConfig:
    """Which signal supplies the training target, kept apart from the loss.

    Bundling the two into one name makes combinations inexpressible, such as
    training the frontier distance under a regression loss, so they are two
    fields that vary independently.
    """

    family: str = "frontier_hard"
    horizon: int = 0
    decay: str = "exponential"
    decay_length: float = 256.0
    signal: str = "repetition_score"

    def __post_init__(self) -> None:
        from degeneration_probe.data.labels import DECAY_SHAPES, LABEL_FAMILIES, TOKEN_SIGNALS

        self.decay_length = _as_float(self.decay_length)
        if self.family not in LABEL_FAMILIES:
            raise ValueError(f"training.label.family must be one of {sorted(LABEL_FAMILIES)}")
        if self.horizon < 0:
            raise ValueError("training.label.horizon must be non-negative")
        if self.decay not in DECAY_SHAPES:
            raise ValueError(f"training.label.decay must be one of {sorted(DECAY_SHAPES)}")
        if self.decay_length <= 0:
            raise ValueError("training.label.decay_length must be positive")
        if self.signal not in TOKEN_SIGNALS:
            raise ValueError(f"training.label.signal must be one of {sorted(TOKEN_SIGNALS)}")


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
    """Where a run writes, and how much of its state survives an interruption."""

    root_dir: str = "outputs"
    strategy: str = "epoch"
    steps: Optional[int] = None
    save_total_limit: int = 2
    keep_best: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    resume: str = "auto"

    def __post_init__(self) -> None:
        if self.strategy not in {"no", "steps", "epoch"}:
            raise ValueError("training.checkpoint.strategy must be no, steps, or epoch")
        if self.strategy == "steps" and not self.steps:
            raise ValueError("training.checkpoint.steps is required when strategy='steps'")
        if self.keep_best and self.strategy == "no":
            raise ValueError(
                "training.checkpoint.keep_best requires a save strategy: with strategy='no' "
                "there is no checkpoint to select from"
            )
        if self.resume not in {"auto", "never"} and not Path(self.resume).is_absolute():
            raise ValueError(
                "training.checkpoint.resume must be 'auto', 'never', or an absolute checkpoint path"
            )


@dataclass
class WandbConfig:
    """Tracking destination and the labels a run is filtered by.

    ``name`` and ``group`` default to values derived from the run's axes, so a
    sweep produces readable, non-colliding runs without hand-written names.
    ``tags`` are extra labels appended to the derived axis tags.
    """

    enabled: bool = True
    mode: str = "online"
    entity: Optional[str] = None
    project: str = "degeneration-probes"
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    resume: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("training.wandb.mode must be online, offline, or disabled")


@dataclass
class TrainingConfig:
    task: str
    short_name: str
    probe: ProbeConfig
    lora: LoraConfig = field(default_factory=LoraConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
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
            "label": LabelConfig,
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
        from degeneration_probe.data.labels import produces_binary_targets

        if self.loss.name == "bce" and self.loss.bce.use_pos_weight and not produces_binary_targets(
            self.label
        ):
            raise ValueError(
                f"training.loss.bce.use_pos_weight has no meaning for label family "
                f"{self.label.family!r}: a positive rate is only defined when every target is 0 or 1"
            )
        if self.checkpoint.keep_best:
            # Selecting the best checkpoint means saving exactly when validating.
            if self.validation.strategy != self.checkpoint.strategy or (
                self.validation.strategy == "steps"
                and self.validation.steps != self.checkpoint.steps
            ):
                raise ValueError(
                    "training.checkpoint.keep_best requires the checkpoint and validation "
                    "cadences to match, so that every saved checkpoint has a score"
                )


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
    # Left unset so that one seed drives weight init, batch order and sampling
    # alike; a seed repeat is then a single override.
    seed: Optional[int] = None

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
        if self.dataset.sampling.seed is None:
            self.dataset.sampling.seed = self.training.runtime.seed

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ExperimentConfig":
        unexpected = set(config) - {"model", "training", "dataset"}
        if unexpected:
            raise ValueError(f"Unexpected top-level config sections: {sorted(unexpected)}")
        return cls(**config)
