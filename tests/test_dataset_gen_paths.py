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
APERTUS_INSTRUCT_YAML = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"
APERTUS1P5_CAPFILTER_YAML = (
    REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus1p5-capfilter-linear-it8816.yaml"
)
APERTUS1P5_SFT256K_YAML = (
    REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus1p5-sft256k-4200.yaml"
)
ALL_DATASET_YAMLS = [APERTUS_INSTRUCT_YAML, APERTUS1P5_CAPFILTER_YAML, APERTUS1P5_SFT256K_YAML]


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


def test_labels_shard_path_in_domain_source(config):
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
    assert paths_module.is_held_out_domain(config, "aime_2025") is False
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


def test_apertus_instruct_yaml_matches_config_schema():
    loaded = DatasetGenConfig.from_yaml(APERTUS_INSTRUCT_YAML)

    assert loaded.model_name == "swiss-ai/Apertus-8B-Instruct-2509"
    assert len(loaded.in_domain_sources) == 5
    assert {src["name"]: src["n_prompts"] for src in loaded.in_domain_sources} == {
        "deepmath_103k": 600,
        "numinamath_1_5": 600,
        "if_sft_data_verified": 600,
        "llama_nemotron": 600,
        "aime_2025": 30,
    }
    assert len(loaded.held_out_sources) == 2
    assert {src["name"]: src["n_prompts"] for src in loaded.held_out_sources} == {
        "medical_o1": 600,
        "codeforces": 600,
    }
    assert loaded.max_new_tokens == 4096
    assert loaded.split_fractions == {"train": 0.70, "val": 0.15, "test_indomain": 0.15}


@pytest.mark.parametrize("yaml_path", ALL_DATASET_YAMLS, ids=lambda p: p.stem)
def test_dataset_yaml_loads_and_points_at_its_own_output_root(yaml_path):
    """Every one of the three dataset configs must load, and each must write
    to a distinct output_root/work_root named after that same dataset (this
    was the actual bug class the full_scale -> degeneration-dataset-*
    rename risked: a copy-pasted config silently pointing at the wrong, or
    another dataset's, storage location)."""
    loaded = DatasetGenConfig.from_yaml(yaml_path)
    assert loaded.output_root.name == yaml_path.stem
    assert loaded.work_root.name == f"{yaml_path.stem}_work"


def test_all_three_datasets_share_identical_sampling_params():
    """The three datasets are meant to be directly comparable -- same prompt
    sample, same rollout budget -- differing only in which model produced
    the completions (model_name/tokenizer_name) and where they're stored
    (output_root/work_root)."""
    configs = [DatasetGenConfig.from_yaml(p) for p in ALL_DATASET_YAMLS]
    comparable_fields = [
        "in_domain_sources",
        "held_out_sources",
        "n_rollouts_per_prompt",
        "max_new_tokens",
        "temperature",
        "top_p",
        "seed",
        "split_fractions",
    ]
    baseline = configs[0]
    for other in configs[1:]:
        for field_name in comparable_fields:
            assert getattr(other, field_name) == getattr(baseline, field_name), field_name

    model_names = {c.model_name for c in configs}
    assert len(model_names) == 3, "each dataset must point at a distinct model"


def test_source_missing_required_key_raises():
    with pytest.raises(ValueError):
        DatasetGenConfig(in_domain_sources=[{"name": "bad_source"}])


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        DatasetGenConfig(split_fractions={"train": 0.5, "val": 0.1, "test_indomain": 0.1})
