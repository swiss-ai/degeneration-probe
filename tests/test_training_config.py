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
