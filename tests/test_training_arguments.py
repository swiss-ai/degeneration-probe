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
    # Validation runs on a step cadence, and saving matches it: a metric that
    # picks a step is useless if no weights were written there.
    assert args.save_strategy.value == "steps"
    assert args.eval_strategy.value == "steps"
    assert args.save_steps == args.eval_steps
    assert args.load_best_model_at_end is True
    # Every checkpoint is kept. They hold one small head per depth, and keeping
    # them is what lets a selection rule be reconsidered without retraining.
    assert args.save_total_limit is None
    # Selection keys off an operating point, not the loss and not a ranking. A
    # token loss is dominated by the trivially separable in-pattern tokens, and
    # a ranking metric reaches its ceiling while the probe is still improving,
    # after which it can no longer tell two checkpoints apart.
    assert args.metric_for_best_model == "recall_at_budget"
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


def test_a_batch_that_overshoots_the_budget_is_refused_when_it_can_be_shrunk():
    """A step cannot be smaller than one micro-batch, so it silently overshoots.

    Left alone this hands the widest window several times the tokens of every
    recipe it is compared with, which is invisible in the results and lands on
    exactly the setting the comparison exists to test. A batch of more than one
    example can be made smaller, so that case is refused with the two settings
    that would fix it named.
    """
    from degeneration_probe.training.arguments import resolve_token_budget

    with pytest.raises(ValueError, match="cannot come in under the requested"):
        resolve_token_budget(
            2048, valid_tokens=512_000, examples=1000, per_device_batch_size=8
        )

    # Halving the batch brings one micro-batch inside the budget exactly.
    resolved = resolve_token_budget(
        2048, valid_tokens=512_000, examples=1000, per_device_batch_size=4
    )
    assert resolved["tokens_per_step_realized"] == 2048
    assert resolved["gradient_accumulation_steps"] == 1


def test_every_window_in_a_sweep_lands_on_the_same_budget():
    """The point of the budget is that only the choice of tokens differs."""
    from degeneration_probe.training.arguments import resolve_token_budget

    for window in (64, 128, 256, 512):
        resolved = resolve_token_budget(
            4096,
            valid_tokens=window * 1000,
            examples=1000,
            per_device_batch_size=8,
        )
        assert resolved["tokens_per_step_realized"] == 4096, window
