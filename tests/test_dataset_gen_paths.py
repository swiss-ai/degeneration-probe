"""Tests for degeneration_probe.dataset_gen (config + paths scaffolding).

Loads the dataset_gen modules directly by file path (bypassing
``degeneration_probe/__init__.py``), the same pattern used in
test_repetition_converter.py / test_metrics.py, since the top-level package
``__init__`` pulls in torch/transformers/peft which aren't required for this
lightweight, no-model-loading corner of the codebase.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GEN_DIR = REPO_ROOT / "degeneration_probe" / "dataset_gen"
PILOT_V1_YAML = REPO_ROOT / "configs" / "dataset_gen" / "pilot_v1.yaml"


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dataset_gen():
    """Load config.py and paths.py without importing the real degeneration_probe package."""
    degeneration_probe_pkg = types.ModuleType("degeneration_probe")
    degeneration_probe_pkg.__path__ = [str(REPO_ROOT / "degeneration_probe")]
    sys.modules.setdefault("degeneration_probe", degeneration_probe_pkg)

    dataset_gen_pkg = types.ModuleType("degeneration_probe.dataset_gen")
    dataset_gen_pkg.__path__ = [str(DATASET_GEN_DIR)]
    sys.modules.setdefault("degeneration_probe.dataset_gen", dataset_gen_pkg)

    config_module = _load_module(
        "degeneration_probe.dataset_gen.config", DATASET_GEN_DIR / "config.py"
    )
    paths_module = _load_module(
        "degeneration_probe.dataset_gen.paths", DATASET_GEN_DIR / "paths.py"
    )
    return config_module, paths_module


config_module, paths_module = _load_dataset_gen()
PilotDatasetConfig = config_module.PilotDatasetConfig


@pytest.fixture
def config() -> "PilotDatasetConfig":
    return PilotDatasetConfig(
        output_root="/tmp/dataset_v2_pilot",
        work_root="/tmp/dataset_v2_pilot_work",
    )


def test_prompts_path(config):
    assert paths_module.prompts_path(config) == Path(
        "/tmp/dataset_v2_pilot/prompts/prompts.parquet"
    )


def test_manifest_path(config):
    assert paths_module.manifest_path(config) == Path("/tmp/dataset_v2_pilot/manifest.json")


def test_generations_shard_path(config):
    got = paths_module.generations_shard_path(config, "in_domain", 3)
    assert got == Path("/tmp/dataset_v2_pilot/generations/in_domain/shard_00003.parquet")


def test_labels_shard_path_held_out(config):
    got = paths_module.labels_shard_path(config, "held_out", 0)
    assert got == Path("/tmp/dataset_v2_pilot/labels/held_out/shard_00000.parquet")


def test_rollout_activation_path(config):
    got = paths_module.rollout_activation_path(config, "in_domain", "deepmath_103k_0007", 2)
    assert got == Path(
        "/tmp/dataset_v2_pilot/activations/in_domain/deepmath_103k_0007/rollout_2.safetensors"
    )


def test_activations_manifest_path(config):
    assert paths_module.activations_manifest_path(config) == Path(
        "/tmp/dataset_v2_pilot/activations/manifest.parquet"
    )


def test_split_path(config):
    assert paths_module.split_path(config, "train") == Path(
        "/tmp/dataset_v2_pilot/splits/train.jsonl"
    )


def test_invalid_domain_rejected(config):
    with pytest.raises(ValueError):
        paths_module.generations_dir(config, "bogus_domain")


def test_all_output_dirs_are_under_output_root(config):
    dirs = paths_module.all_output_dirs(config)
    assert all(str(d).startswith("/tmp/dataset_v2_pilot") for d in dirs)
    # sanity: both domains represented for the per-domain dirs
    assert any("in_domain" in str(d) for d in dirs)
    assert any("held_out" in str(d) for d in dirs)


def test_config_yaml_round_trip(tmp_path):
    original = PilotDatasetConfig(
        n_rollouts_per_prompt=5,
        max_new_tokens=256,
        output_root=str(tmp_path / "out"),
        work_root=str(tmp_path / "work"),
    )
    yaml_path = tmp_path / "config.yaml"
    original.to_yaml(yaml_path)

    loaded = PilotDatasetConfig.from_yaml(yaml_path)

    assert loaded.to_dict() == original.to_dict()
    assert loaded.n_rollouts_per_prompt == 5
    assert loaded.max_new_tokens == 256


def test_pilot_v1_yaml_matches_config_schema():
    loaded = PilotDatasetConfig.from_yaml(PILOT_V1_YAML)

    assert loaded.model_name == "swiss-ai/Apertus-8B-Instruct-2509"
    assert len(loaded.in_domain_sources) == 4
    assert all(src["n_prompts"] == 70 for src in loaded.in_domain_sources)
    assert len(loaded.held_out_sources) == 2
    assert all(src["n_prompts"] == 60 for src in loaded.held_out_sources)
    assert loaded.split_fractions == {"train": 0.70, "val": 0.15, "test_indomain": 0.15}


def test_source_missing_required_key_raises():
    with pytest.raises(ValueError):
        PilotDatasetConfig(in_domain_sources=[{"name": "bad_source"}])


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        PilotDatasetConfig(split_fractions={"train": 0.5, "val": 0.1, "test_indomain": 0.1})
