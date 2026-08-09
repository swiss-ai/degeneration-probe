from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.training.run_identity import (
    config_fingerprint,
    derive_group_name,
    derive_run_name,
    derive_tags,
    run_axes,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _resolved(overrides=None) -> dict:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="main", overrides=overrides or [])
    return OmegaConf.to_container(cfg, resolve=True)


def _experiment(overrides=None) -> ExperimentConfig:
    return ExperimentConfig.from_dict(_resolved(overrides))


def test_run_name_carries_the_axes_a_reader_scans_for():
    name = derive_run_name(_experiment())
    for fragment in ("apertus-8b-instruct", "L30", "bce", "lora-all", "s42"):
        assert fragment in name


def test_seed_repeats_share_a_group_but_not_a_name():
    first = _experiment(["training.runtime.seed=1"])
    second = _experiment(["training.runtime.seed=2"])
    assert derive_run_name(first) != derive_run_name(second)
    assert derive_group_name(first) == derive_group_name(second)


def test_a_learning_relevant_change_moves_the_fingerprint():
    baseline = _experiment()
    changed = _experiment(["training.optimizer.probe_learning_rate=3e-4"])
    assert config_fingerprint(baseline) != config_fingerprint(changed)
    assert derive_run_name(baseline) != derive_run_name(changed)


def test_observation_settings_leave_the_fingerprint_alone():
    baseline = _experiment()
    observed = _experiment(
        [
            "training.runtime.logging_steps=1",
            "training.checkpoint.save_total_limit=5",
            "training.wandb.notes=anything",
        ]
    )
    assert config_fingerprint(baseline) == config_fingerprint(observed)


def test_tags_are_filterable_key_value_pairs_plus_the_configured_extras():
    config = _experiment(["+training.wandb.tags=[sweep:layer]"])
    tags = derive_tags(config)
    assert "layer:30" in tags
    assert "loss:bce" in tags
    assert "lora:all" in tags
    assert "seed:42" in tags
    assert "sweep:layer" in tags
    assert len(tags) == len(set(tags))


def test_explicit_names_win_over_the_derived_ones():
    resolved = _resolved()
    resolved["training"]["wandb"]["name"] = "handpicked"
    resolved["training"]["wandb"]["group"] = "handpicked-group"
    config = ExperimentConfig.from_dict(resolved)
    assert derive_run_name(config) == "handpicked"
    assert derive_group_name(config) == "handpicked-group"


def test_lora_scope_appears_in_the_axes_and_collapses_when_disabled():
    disabled = _experiment(["training.lora.enabled=false"])
    axes = run_axes(disabled)
    assert axes["lora"] == "none"
    assert axes["lora_rank"] == 0
    assert "lora-none" in derive_run_name(disabled)


def test_one_seed_drives_weight_init_and_sampling_alike():
    config = _experiment(["training.runtime.seed=7"])
    assert config.dataset.sampling.seed == 7


def test_an_explicit_sampling_seed_is_still_honoured():
    resolved = _resolved()
    resolved["dataset"]["sampling"]["seed"] = 123
    config = ExperimentConfig.from_dict(resolved)
    assert config.dataset.sampling.seed == 123


def test_best_checkpoint_selection_requires_matching_cadences():
    resolved = _resolved()
    resolved["training"]["checkpoint"]["strategy"] = "steps"
    resolved["training"]["checkpoint"]["steps"] = 50
    with pytest.raises(ValueError, match="cadences to match"):
        ExperimentConfig.from_dict(deepcopy(resolved))
