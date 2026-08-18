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
    # Several depths at once. Reading one layer out of the cache costs almost
    # what reading every layer costs, since the seek dominates, so a probe per
    # depth can be trained in a single pass instead of one pass each. The heads
    # share no parameters and their losses are added, which makes each one's
    # gradient identical to what it would be trained alone.
    layers: Optional[List[int]] = None
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
        if self.layers is not None:
            self.layers = [int(layer) for layer in self.layers]
            if not self.layers:
                raise ValueError("training.probe.layers must name at least one layer")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("training.probe.layers must not repeat a layer")
            if any(layer < 0 for layer in self.layers):
                raise ValueError("training.probe.layers must all be non-negative")

    @property
    def probed_layers(self) -> List[int]:
        """Every depth this run trains a head for."""
        return list(self.layers) if self.layers is not None else [self.layer]


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
class FeaturesConfig:
    """Where the probe's hidden states come from.

    ``adapted`` runs the language model, which adapters require since they
    change the representation at every step. ``cached`` reads states computed
    once with the model frozen, which removes the model from training entirely
    and is what makes a large sweep affordable.
    """

    regime: str = "adapted"

    def __post_init__(self) -> None:
        if self.regime not in {"adapted", "cached"}:
            raise ValueError("training.features.regime must be adapted or cached")


@dataclass
class SelectionConfig:
    """Which tokens of a rollout a run learns from.

    Distinct from ``dataset.sampling``, which decides which rollouts enter the
    pool at all: this decides what is taken from each of them. The rules form a
    ladder in which each rung changes exactly one decision, so a difference
    between adjacent rungs is attributable to that decision.
    """

    strategy: str = "all_tokens"
    window_size: int = 128
    anchor: str = "trailing"
    positive_fraction: float = 0.25
    max_rollouts_per_prompt: Optional[int] = None
    hard_negative_fraction: float = 0.5
    resample_each_epoch: bool = True

    def __post_init__(self) -> None:
        from degeneration_probe.data.sampling import ANCHORS, STRATEGIES

        self.positive_fraction = _as_float(self.positive_fraction)
        self.hard_negative_fraction = _as_float(self.hard_negative_fraction)
        if self.strategy not in STRATEGIES:
            raise ValueError(f"training.selection.strategy must be one of {sorted(STRATEGIES)}")
        if self.anchor not in ANCHORS:
            raise ValueError(f"training.selection.anchor must be one of {sorted(ANCHORS)}")
        if self.window_size < 1:
            raise ValueError("training.selection.window_size must be at least 1")
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError("training.selection.positive_fraction must lie strictly in (0, 1)")
        if not 0.0 <= self.hard_negative_fraction <= 1.0:
            raise ValueError("training.selection.hard_negative_fraction must lie in [0, 1]")


@dataclass
class StoppingConfig:
    """When a depth has stopped improving, and which checkpoint it is judged on.

    The same four numbers the replay of saved checkpoints is read with, so a run
    that stops itself and a run that is judged afterwards are judged by one
    rule. They set how long a run trains far more than they set what it
    reports: over the whole range that has been measured, the depth a run is
    reported at and the value it reports barely move, while the step it stops at
    roughly doubles. Read them as a budget, not as an accuracy setting.
    """

    enabled: bool = True
    # Coverage inside the loop a depth must reach before its checkpoints can be
    # selected from. There is no natural break in this quantity, so the value is
    # a judgement: it separates depths that never found the loop from the rest.
    floor: float = 0.3
    # Which run-up width the objective is measured over.
    band: int = 256
    # How much better a step has to be to count as an improvement. Roughly the
    # sampling error of the objective on a full validation split, and several
    # times the step-to-step wobble between neighbouring checkpoints.
    tolerance: float = 0.002
    # Evaluations without an improvement before a depth stops. Counted in
    # evaluations, so it has to move with the validation cadence.
    patience: int = 4

    def __post_init__(self) -> None:
        from degeneration_probe.evaluation.protocol import WARNING_BANDS

        self.floor = _as_float(self.floor)
        self.tolerance = _as_float(self.tolerance)
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("training.stopping.floor must lie in [0, 1]")
        if self.band not in WARNING_BANDS:
            raise ValueError(
                f"training.stopping.band must be one of {list(WARNING_BANDS)}: the objective "
                "is only measured at the widths validation reports"
            )
        if self.tolerance < 0:
            raise ValueError("training.stopping.tolerance must be non-negative")
        if self.patience < 1:
            raise ValueError("training.stopping.patience must be at least 1")

    def as_rule(self):
        """The rule itself, as the replay and the analysis build it."""
        from degeneration_probe.evaluation.head_selection import StoppingRule

        return StoppingRule(
            floor=self.floor,
            band=self.band,
            tolerance=self.tolerance,
            patience=self.patience,
        )


@dataclass
class BudgetConfig:
    """The training budget, in the units that make recipes comparable.

    An epoch is not a fixed amount of learning here: under the exhaustive rule
    it is the whole corpus, under an anchored-window rule it is one window per
    rollout. Training each recipe for "one epoch" would hand them budgets that
    differ by an order of magnitude, and the measured difference would be mostly
    a difference in training length. Fixing steps and tokens per step instead is
    what makes a comparison between rules mean anything.
    """

    tokens_per_step: Optional[int] = None
    patience: Optional[int] = None
    collapse_threshold: float = 0.01

    def __post_init__(self) -> None:
        self.collapse_threshold = _as_float(self.collapse_threshold)
        if self.tokens_per_step is not None and self.tokens_per_step < 1:
            raise ValueError("training.budget.tokens_per_step must be positive")
        if self.patience is not None:
            # Kept only so that a run recorded before the rule existed still
            # loads. It stopped on whichever scalar best-model selection was
            # keyed to, which is a different question from the one the rule
            # asks, so pointing it at the rule silently would change what a
            # configuration means.
            raise ValueError(
                "training.budget.patience has been replaced by training.stopping.patience, "
                "which counts evaluations without an improvement in run-up coverage rather "
                "than in whichever scalar best-model selection happens to track"
            )
        if self.collapse_threshold < 0:
            raise ValueError("training.budget.collapse_threshold must be non-negative")


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
    # How many rollouts the monitoring split is cut down to, keeping every
    # positive. Training reads a short window per rollout, but evaluation reads
    # whole ones, so a probe covering many depths pays for every token of every
    # layer each time it is monitored. A curve does not need the whole split;
    # the final numbers do, and those are measured once at the end.
    max_rollouts: Optional[int] = None
    # Which splits the end-of-run evaluation covers. Left unset it scores all of
    # them, which is rarely what an exploratory run wants: the test splits are
    # far larger than the monitor and are not read until a recipe is chosen.
    final_splits: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.strategy not in {"no", "steps", "epoch"}:
            raise ValueError("training.validation.strategy must be no, steps, or epoch")
        if self.strategy == "steps" and not self.steps:
            raise ValueError("training.validation.steps is required when strategy='steps'")
        if self.max_rollouts is not None and self.max_rollouts < 1:
            raise ValueError("training.validation.max_rollouts must be positive")


@dataclass
class CheckpointConfig:
    """Where a run writes, and how much of its state survives an interruption."""

    root_dir: str = "outputs"
    strategy: str = "epoch"
    steps: Optional[int] = None
    save_total_limit: int = 2
    keep_best: bool = True
    # The rule's objective, gated on coverage inside the loop. Rollout-level
    # recall saturates: on a split with a hundred degenerate answers it reads
    # the same at every step of a run, so the checkpoint it picks is whichever
    # one happened to catch one more answer.
    metric_for_best_model: str = "selection_score"
    greater_is_better: bool = True
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
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
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
            "features": FeaturesConfig,
            "selection": SelectionConfig,
            "budget": BudgetConfig,
            "stopping": StoppingConfig,
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
        if self.features.regime == "cached" and self.lora.enabled and self.lora.layers != "none":
            raise ValueError(
                "training.features.regime='cached' cannot train adapters: the cached states "
                "describe the unadapted model, so training against them would fit a "
                "representation that no longer exists. Set training.lora.enabled=false."
            )
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
    # Where the build's cached hidden states live. Defaults to the
    # ``activations/`` subtree of the build itself; set it only for a build
    # whose activations were written elsewhere because they were too large to
    # sit beside it. Must agree with the same key in the build's own
    # generation config, which is what put them there.
    activations_root: Optional[str] = None

    @property
    def activations_dir(self) -> Path:
        """The directory holding ``<domain>/<prompt_id>/rollout_<k>.safetensors``."""
        if self.activations_root is not None:
            return Path(self.activations_root)
        return Path(self.build_root) / "activations"

    @property
    def activations_manifest_path(self) -> Path:
        return self.activations_dir / "manifest.parquet"

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
