from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.evaluation.metrics import build_validation_metrics


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(overrides=None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="main", overrides=overrides or [])


def test_main_composes_only_model_training_and_dataset():
    cfg = _compose()
    assert set(cfg.keys()) == {"model", "training", "dataset"}
    assert cfg.training.task == "degeneration"
    assert cfg.training.loss.name == "bce"
    assert cfg.training.probe.layer == 30
    assert "wandb" in cfg.training
    assert "wandb" not in cfg


def test_mse_override_keeps_the_same_task_and_probe():
    cfg = _compose(["training.loss.name=mse"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    experiment = ExperimentConfig.from_dict(resolved)
    assert experiment.training.task == "degeneration"
    assert experiment.training.loss.name == "mse"
    assert experiment.training.probe.layer == 30


def test_only_degeneration_task_is_accepted():
    cfg = OmegaConf.to_container(_compose(), resolve=True)
    cfg["training"]["task"] = "legacy"
    with pytest.raises(ValueError, match="only supported task"):
        ExperimentConfig.from_dict(cfg)


def test_validation_metric_registry_starts_empty_and_rejects_unknown_names():
    assert build_validation_metrics([]) == {}
    with pytest.raises(ValueError, match="Unknown validation metric"):
        build_validation_metrics(["not_registered"])


def test_monitoring_can_be_cut_down_without_losing_a_positive():
    """Thinning the monitor must never cost a positive rollout.

    Positives are scarce and the rank metric is built from them, so dropping
    any would make the curve noisier while saving almost nothing: the negatives
    are what the split is mostly made of.
    """
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_probe import subsample_for_monitoring

    records = [SimpleNamespace(is_positive=i % 10 == 0) for i in range(200)]
    dataset = SimpleNamespace(records=records)

    reduced = subsample_for_monitoring(dataset, 50, seed=1)
    assert len(reduced.records) == 50
    assert sum(r.is_positive for r in reduced.records) == 20
    # Deterministic, so the monitor is the same population at every evaluation.
    assert [id(r) for r in reduced.records] == [
        id(r) for r in subsample_for_monitoring(dataset, 50, seed=1).records
    ]
    # The original is untouched, since the full split is still evaluated at the end.
    assert len(dataset.records) == 200


def test_no_limit_leaves_the_monitor_alone():
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_probe import subsample_for_monitoring

    dataset = SimpleNamespace(records=[SimpleNamespace(is_positive=False)] * 12)
    assert subsample_for_monitoring(dataset, None, seed=1) is dataset
    assert subsample_for_monitoring(dataset, 99, seed=1) is dataset


def test_the_stopping_block_builds_the_same_rule_the_replay_uses():
    """One rule, whether a run stops itself or is judged after the fact."""
    from degeneration_probe.config import StoppingConfig
    from degeneration_probe.evaluation.head_selection import StoppingRule

    config = StoppingConfig(floor=0.2, band=128, tolerance="0.004", patience=3)
    assert config.as_rule() == StoppingRule(
        floor=0.2, band=128, tolerance=0.004, patience=3
    )
    assert config.as_rule().objective == "warning_recall_128"


def test_a_width_validation_never_measures_is_refused():
    """The objective can only be read at the widths the record carries."""
    from degeneration_probe.config import StoppingConfig

    with pytest.raises(ValueError, match="training.stopping.band"):
        StoppingConfig(band=200)


@pytest.mark.parametrize(
    "kwargs", [{"floor": 1.5}, {"tolerance": -0.1}, {"patience": 0}]
)
def test_the_stopping_block_refuses_settings_that_cannot_decide_anything(kwargs):
    from degeneration_probe.config import StoppingConfig

    with pytest.raises(ValueError):
        StoppingConfig(**kwargs)


def test_the_old_patience_setting_says_what_replaced_it():
    """It stopped on whichever scalar best-model selection tracked, which is a
    different question from the one the rule asks, so it cannot be repointed
    silently."""
    from degeneration_probe.config import BudgetConfig

    assert BudgetConfig().patience is None
    with pytest.raises(ValueError, match="training.stopping.patience"):
        BudgetConfig(patience=4)
