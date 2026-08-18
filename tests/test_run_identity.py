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


def test_a_run_reading_every_depth_is_not_named_after_one():
    """Naming a many-headed run after a single layer states something untrue.

    The single-layer setting stays at whatever it was when the run asked for
    every depth instead, so a name built from it would claim a depth the run
    never committed to, and the claim would follow the run into its directory
    name, its group, and every table built from either.
    """
    every = _experiment(["training.probe.layers=[1,2,3]", "training.probe.layer=30"])
    name = derive_run_name(every)
    assert "L1-3" in name
    assert "L30" not in name

    axes = run_axes(every)
    assert axes["layer"] is None, "a run over many depths has no one layer"
    assert axes["layers"] == "1-3"

    # A run that really does read one depth is unchanged by any of this.
    one = _experiment(["training.probe.layer=12"])
    assert "L12" in derive_run_name(one)
    assert run_axes(one)["layer"] == 12
    assert run_axes(one)["layers"] is None


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
    """Keeping the best checkpoint needs a checkpoint at every step it judges.

    Saving less often than validating means the step a metric picked may have no
    weights behind it, and the run silently keeps a neighbour instead.
    """
    resolved = _resolved()
    assert resolved["training"]["validation"]["steps"] == 50, "the default cadence moved"
    resolved["training"]["checkpoint"]["steps"] = 200
    with pytest.raises(ValueError, match="cadences to match"):
        ExperimentConfig.from_dict(deepcopy(resolved))

    resolved["training"]["checkpoint"]["strategy"] = "epoch"
    with pytest.raises(ValueError, match="cadences to match"):
        ExperimentConfig.from_dict(deepcopy(resolved))


def test_when_a_run_stops_is_not_part_of_what_it_is():
    """Two runs differing only in when they stop are one run truncated twice.

    Every checkpoint is kept, so the shorter run's trajectory is a prefix of the
    longer one's. Sharing an identity is what lets a rule given more patience
    continue a run rather than repeat it, and what keeps a re-reading of the
    same saved checkpoints from looking like a different experiment.
    """
    base = _experiment()
    patient = _experiment()
    patient.training.stopping.patience = 12
    patient.training.stopping.floor = 0.1
    patient.training.stopping.enabled = False
    assert derive_run_name(base) == derive_run_name(patient)
    assert config_fingerprint(base) == config_fingerprint(patient)


def test_what_a_run_learns_from_is_part_of_what_it_is():
    """The guard on the test above: the fingerprint still separates recipes."""
    base = _experiment()
    other = _experiment()
    other.training.selection.window_size = base.training.selection.window_size * 2
    assert config_fingerprint(base) != config_fingerprint(other)


def test_a_setting_left_alone_does_not_enter_the_fingerprint():
    """The failure this prevents: adding a field renames every existing run.

    A run derives its own name and then looks for a directory under it, so a
    renamed run finds nothing to continue and silently starts from zero. A field
    nobody set records no decision, so it must not move the hash.
    """
    resolved = _resolved()
    # A configuration written before the field existed, against one written
    # after it and left at its default.
    without = deepcopy(resolved)
    without["dataset"].pop("activations_root", None)
    with_default = deepcopy(resolved)
    with_default["dataset"]["activations_root"] = None
    a = ExperimentConfig.from_dict(without)
    b = ExperimentConfig.from_dict(with_default)
    assert config_fingerprint(a) == config_fingerprint(b)
    assert derive_run_name(a) == derive_run_name(b)


def test_a_setting_actually_chosen_does_enter_it():
    """The guard on the test above: the hash still separates real decisions."""
    resolved = _resolved()
    chosen = deepcopy(resolved)
    chosen["dataset"]["activations_root"] = "/somewhere/else"
    assert config_fingerprint(ExperimentConfig.from_dict(resolved)) != config_fingerprint(
        ExperimentConfig.from_dict(chosen)
    )


def test_setting_a_field_to_its_own_default_changes_nothing():
    """Naming a default explicitly is a way of writing, not a different run."""
    resolved = _resolved()
    spelled_out = deepcopy(resolved)
    spelled_out["training"]["probe"]["normalization"] = "layernorm"
    spelled_out["training"]["lora"]["rank"] = 16
    assert config_fingerprint(ExperimentConfig.from_dict(resolved)) == config_fingerprint(
        ExperimentConfig.from_dict(spelled_out)
    )
