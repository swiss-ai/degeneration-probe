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
FULL_SCALE_YAML = REPO_ROOT / "configs" / "dataset" / "full_scale.yaml"


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
DatasetGenConfig = config_module.DatasetGenConfig


@pytest.fixture
def config() -> "DatasetGenConfig":
    return DatasetGenConfig(
        output_root="/tmp/dataset_gen_test",
        work_root="/tmp/dataset_gen_test_work",
    )


def test_prompts_path(config):
    assert paths_module.prompts_path(config) == Path(
        "/tmp/dataset_gen_test/prompts/prompts.parquet"
    )


def test_manifest_path(config):
    assert paths_module.manifest_path(config) == Path("/tmp/dataset_gen_test/manifest.json")


def test_generations_shard_path(config):
    got = paths_module.generations_shard_path(config, "deepmath_103k", 3)
    assert got == Path("/tmp/dataset_gen_test/generations/deepmath_103k/shard_00003.parquet")


def test_labels_shard_path_held_out_source(config):
    got = paths_module.labels_shard_path(config, "aime_2025", 0)
    assert got == Path("/tmp/dataset_gen_test/labels/aime_2025/shard_00000.parquet")


def test_rollout_activation_path(config):
    got = paths_module.rollout_activation_path(config, "deepmath_103k", "deepmath_103k_0007", 2)
    assert got == Path(
        "/tmp/dataset_gen_test/activations/deepmath_103k/deepmath_103k_0007/rollout_2.safetensors"
    )


def test_activations_manifest_path(config):
    assert paths_module.activations_manifest_path(config) == Path(
        "/tmp/dataset_gen_test/activations/manifest.parquet"
    )


def test_split_path(config):
    assert paths_module.split_path(config, "train") == Path(
        "/tmp/dataset_gen_test/splits/train.jsonl"
    )


def test_invalid_domain_rejected(config):
    with pytest.raises(ValueError):
        paths_module.generations_dir(config, "bogus_domain")


def test_in_domain_and_held_out_split_names_are_rejected_as_domains(config):
    # "in_domain"/"held_out" are the coarse split names, not source names --
    # they must NOT be accepted as a `domain` (that was the bug being fixed).
    with pytest.raises(ValueError):
        paths_module.generations_dir(config, "in_domain")
    with pytest.raises(ValueError):
        paths_module.generations_dir(config, "held_out")


def test_configured_domain_names(config):
    assert paths_module.configured_domain_names(config) == {
        "deepmath_103k",
        "numinamath_1_5",
        "if_sft_data_verified",
        "llama_nemotron",
        "medical_o1",
        "aime_2025",
    }


def test_is_held_out_domain(config):
    assert paths_module.is_held_out_domain(config, "aime_2025") is True
    assert paths_module.is_held_out_domain(config, "medical_o1") is True
    assert paths_module.is_held_out_domain(config, "deepmath_103k") is False
    with pytest.raises(ValueError):
        paths_module.is_held_out_domain(config, "bogus_domain")


def test_all_output_dirs_are_under_output_root(config):
    dirs = paths_module.all_output_dirs(config)
    assert all(str(d).startswith("/tmp/dataset_gen_test") for d in dirs)

    # one subfolder per actual source name (6 total) under generations/,
    # labels/, and activations/ -- not one shared folder per split.
    for domain in paths_module.configured_domain_names(config):
        assert paths_module.generations_dir(config, domain) in dirs
        assert paths_module.labels_dir(config, domain) in dirs
        assert paths_module.activations_domain_dir(config, domain) in dirs

    generations_subdirs = [d for d in dirs if d.parent.name == "generations"]
    labels_subdirs = [d for d in dirs if d.parent.name == "labels"]
    activations_subdirs = [d for d in dirs if d.parent.name == "activations"]
    assert len(generations_subdirs) == 6
    assert len(labels_subdirs) == 6
    assert len(activations_subdirs) == 6

    # the coarse split names must not appear as folders themselves
    assert not any(d.name in ("in_domain", "held_out") for d in dirs)


def test_config_yaml_round_trip(tmp_path):
    original = DatasetGenConfig(
        n_rollouts_per_prompt=5,
        max_new_tokens=256,
        output_root=str(tmp_path / "out"),
        work_root=str(tmp_path / "work"),
    )
    yaml_path = tmp_path / "config.yaml"
    original.to_yaml(yaml_path)

    loaded = DatasetGenConfig.from_yaml(yaml_path)

    assert loaded.to_dict() == original.to_dict()
    assert loaded.n_rollouts_per_prompt == 5
    assert loaded.max_new_tokens == 256


def test_full_scale_yaml_matches_config_schema():
    loaded = DatasetGenConfig.from_yaml(FULL_SCALE_YAML)

    assert loaded.model_name == "swiss-ai/Apertus-8B-Instruct-2509"
    assert len(loaded.in_domain_sources) == 4
    assert all(src["n_prompts"] == 600 for src in loaded.in_domain_sources)
    assert len(loaded.held_out_sources) == 2
    assert {src["name"]: src["n_prompts"] for src in loaded.held_out_sources} == {
        "medical_o1": 600,
        "aime_2025": 30,
    }
    assert loaded.max_new_tokens == 4096
    assert loaded.split_fractions == {"train": 0.70, "val": 0.15, "test_indomain": 0.15}


def test_source_missing_required_key_raises():
    with pytest.raises(ValueError):
        DatasetGenConfig(in_domain_sources=[{"name": "bad_source"}])


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        DatasetGenConfig(split_fractions={"train": 0.5, "val": 0.1, "test_indomain": 0.1})
