from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.training.arguments import build_training_arguments

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _experiment(overrides=None) -> ExperimentConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="main", overrides=overrides or [])
    return ExperimentConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))


def _build(config, tmp_path):
    return build_training_arguments(
        config, run_dir=tmp_path, run_name="a_run", report_to=[]
    )


def test_defaults_produce_resumable_checkpoints_and_keep_the_best(tmp_path):
    args = _build(_experiment(), tmp_path)
    assert args.save_strategy.value == "epoch"
    assert args.eval_strategy.value == "epoch"
    assert args.load_best_model_at_end is True
    assert args.save_total_limit == 2
    # Selection keys off a protocol metric, not the loss: a token loss is
    # dominated by the trivially separable in-pattern tokens.
    assert args.metric_for_best_model == "rollout_auc"
    assert args.greater_is_better is True


def test_a_step_cadence_is_applied_to_both_validation_and_saving(tmp_path):
    config = _experiment(
        [
            "training.validation.strategy=steps",
            "training.validation.steps=25",
            "training.checkpoint.strategy=steps",
            "training.checkpoint.steps=25",
        ]
    )
    args = _build(config, tmp_path)
    assert args.eval_steps == 25
    assert args.save_steps == 25


def test_checkpointing_can_be_switched_off_entirely(tmp_path):
    config = _experiment(
        [
            "training.checkpoint.strategy=no",
            "training.checkpoint.keep_best=false",
            "training.validation.strategy=no",
        ]
    )
    args = _build(config, tmp_path)
    assert args.load_best_model_at_end is False


def test_selection_without_validation_is_rejected_before_a_run_starts():
    with pytest.raises(ValueError, match="cadences to match"):
        _experiment(["training.validation.strategy=no"])


def test_the_token_budget_is_measured_not_assumed():
    """Equal budget has to mean equal tokens, whatever an example holds.

    A rule keeping whole rollouts and a rule emitting short windows differ by an
    order of magnitude per example. Both must land on the same tokens per step,
    which they only do if the size of an example is measured.
    """
    from degeneration_probe.training.arguments import resolve_token_budget

    rollouts = resolve_token_budget(
        20_000, valid_tokens=1_247_000, examples=1_000, per_device_batch_size=1
    )
    windows = resolve_token_budget(
        20_000, valid_tokens=123_000, examples=1_000, per_device_batch_size=8
    )
    assert rollouts["tokens_per_example"] == pytest.approx(1247.0)
    assert windows["tokens_per_example"] == pytest.approx(123.0)
    # Different accumulation, because an example is a different size.
    assert rollouts["gradient_accumulation_steps"] != windows["gradient_accumulation_steps"]
    # Same budget, which is the whole point.
    for resolved in (rollouts, windows):
        assert resolved["tokens_per_step_realized"] == pytest.approx(20_000, rel=0.05)


def test_a_budget_smaller_than_one_example_still_takes_a_step():
    from degeneration_probe.training.arguments import resolve_token_budget

    resolved = resolve_token_budget(
        128, valid_tokens=400_000, examples=100, per_device_batch_size=1
    )
    assert resolved["gradient_accumulation_steps"] == 1
    # Reported honestly as the overshoot it is, rather than as the request.
    assert resolved["tokens_per_step_realized"] == 4000
