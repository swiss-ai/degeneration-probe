"""Tests for degeneration_probe.dataset_gen (config + paths scaffolding).

Loads the dataset_gen modules directly by file path (bypassing
``degeneration_probe/__init__.py``), the same pattern used in
other dataset-generation tests, since this is a lightweight, no-model-loading
corner of the codebase.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GEN_DIR = REPO_ROOT / "degeneration_probe" / "dataset_gen"
BUILD_CONFIG_DIR = REPO_ROOT / "configs" / "dataset" / "builds"
APERTUS_INSTRUCT_YAML = BUILD_CONFIG_DIR / "degeneration-dataset-apertus-8b-instruct.yaml"
ALL_DATASET_YAMLS = sorted(BUILD_CONFIG_DIR.glob("degeneration-dataset-*.yaml"))

# The directly-comparable dataset builds: same prompt sample, same rollout
# budget, differing only in which model produced the completions (see
# notebooks/inspect_dataset.ipynb Section 1). The token budget in particular
# has to match across the family, because a rollout stopped by the cap is what
# the degeneration label keys off -- a build generated to a shorter budget hits
# the cap far more often and for a different reason, so its rates cannot be
# read against the others'.
#
# Named explicitly rather than derived from ALL_DATASET_YAMLS so that an
# unrelated build config can live in the same directory without being swept
# into this family's comparison.
COMPARABLE_FAMILY_YAMLS = [
    APERTUS_INSTRUCT_YAML,
    BUILD_CONFIG_DIR / "degeneration-dataset-apertus1p5-capfilter-linear-it8816.yaml",
    BUILD_CONFIG_DIR / "degeneration-dataset-apertus1p5-sft256k-4200.yaml",
    BUILD_CONFIG_DIR / "degeneration-dataset-llama3p1-8b-instruct.yaml",
    BUILD_CONFIG_DIR / "degeneration-dataset-mistral-7b-instruct-v0p1.yaml",
]


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
        "codeforces",
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

    # one subfolder per actual source name (7 total) under generations/,
    # labels/, and activations/ -- not one shared folder per split.
    for domain in paths_module.configured_domain_names(config):
        assert paths_module.generations_dir(config, domain) in dirs
        assert paths_module.labels_dir(config, domain) in dirs
        assert paths_module.activations_domain_dir(config, domain) in dirs

    generations_subdirs = [d for d in dirs if d.parent.name == "generations"]
    labels_subdirs = [d for d in dirs if d.parent.name == "labels"]
    activations_subdirs = [d for d in dirs if d.parent.name == "activations"]
    assert len(generations_subdirs) == 7
    assert len(labels_subdirs) == 7
    assert len(activations_subdirs) == 7

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


def test_dataset_yaml_points_at_its_own_output_root():
    loaded = DatasetGenConfig.from_yaml(APERTUS_INSTRUCT_YAML)
    assert loaded.output_root.name == APERTUS_INSTRUCT_YAML.stem
    assert loaded.work_root.name == f"{APERTUS_INSTRUCT_YAML.stem}_work"


@pytest.mark.parametrize("yaml_path", ALL_DATASET_YAMLS, ids=lambda path: path.stem)
def test_every_dataset_build_config_loads_and_uses_its_own_roots(yaml_path):
    loaded = DatasetGenConfig.from_yaml(yaml_path)
    assert loaded.output_root.name == yaml_path.stem
    assert loaded.work_root.name == f"{yaml_path.stem}_work"
    if loaded.activations_root is not None:
        assert loaded.activations_root.name.startswith(yaml_path.stem)


def test_activations_default_to_living_inside_the_build():
    config = DatasetGenConfig(output_root="/somewhere/a-build")
    assert paths_module.activations_root(config) == Path("/somewhere/a-build/activations")
    assert paths_module.activations_manifest_path(config) == Path(
        "/somewhere/a-build/activations/manifest.parquet"
    )


def test_a_relocated_activations_root_moves_the_whole_cache_tree_and_nothing_else():
    """Hidden states can outgrow the filesystem holding the rest of a build."""
    config = DatasetGenConfig(output_root="/somewhere/a-build", activations_root="/elsewhere/cache")

    assert paths_module.activations_root(config) == Path("/elsewhere/cache")
    assert paths_module.activations_manifest_path(config) == Path("/elsewhere/cache/manifest.parquet")
    assert paths_module.rollout_activation_path(config, "aime_2025", "aime_2025_00000", 3) == Path(
        "/elsewhere/cache/aime_2025/aime_2025_00000/rollout_3.safetensors"
    )

    # Everything that is small and durable stays with the build itself.
    assert paths_module.prompts_path(config) == Path("/somewhere/a-build/prompts/prompts.parquet")
    assert paths_module.llm_judge_dir(config) == Path("/somewhere/a-build/llm_judge")
    assert paths_module.generations_shard_path(config, "aime_2025", 0) == Path(
        "/somewhere/a-build/generations/aime_2025/shard_00000.parquet"
    )


@pytest.mark.parametrize(
    "activations_root", [None, "/elsewhere/cache"], ids=["in-build", "relocated"]
)
def test_a_cached_rollout_always_sits_under_the_activations_root(activations_root):
    """The activations manifest records each rollout relative to that root.

    Anchoring anywhere else has no answer for a build whose hidden states are
    on a different filesystem than the build directory.
    """
    config = DatasetGenConfig(
        output_root="/somewhere/a-build", activations_root=activations_root
    )
    rollout = paths_module.rollout_activation_path(config, "aime_2025", "aime_2025_00000", 3)
    relative = rollout.relative_to(paths_module.activations_root(config))
    assert relative == Path("aime_2025/aime_2025_00000/rollout_3.safetensors")


def test_all_dataset_builds_keep_identical_sampling_parameters():
    assert len(COMPARABLE_FAMILY_YAMLS) == 5
    configs = [DatasetGenConfig.from_yaml(path) for path in COMPARABLE_FAMILY_YAMLS]
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
    for other in configs[1:]:
        for field_name in comparable_fields:
            assert getattr(other, field_name) == getattr(configs[0], field_name), field_name
    assert len({config.model_name for config in configs}) == 5


def test_source_missing_required_key_raises():
    with pytest.raises(ValueError):
        DatasetGenConfig(in_domain_sources=[{"name": "bad_source"}])


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        DatasetGenConfig(split_fractions={"train": 0.5, "val": 0.1, "test_indomain": 0.1})
