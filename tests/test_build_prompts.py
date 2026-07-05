"""Tests for degeneration_probe.dataset_gen.build_prompts.

Uses tiny fake in-memory (source_row_id, prompt_text) rows / DataFrames --
no network access, no real HF dataset loading. Loads config.py, paths.py,
and build_prompts.py directly by file path (bypassing
``degeneration_probe/__init__.py``), the same pattern used in
test_dataset_gen_paths.py / test_repetition_converter.py, since the
top-level package ``__init__`` pulls in torch/transformers/peft which
aren't required here.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GEN_DIR = REPO_ROOT / "degeneration_probe" / "dataset_gen"


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dataset_gen():
    """Load config.py, paths.py, and build_prompts.py without importing the real package."""
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
    build_prompts_module = _load_module(
        "degeneration_probe.dataset_gen.build_prompts", DATASET_GEN_DIR / "build_prompts.py"
    )
    return config_module, paths_module, build_prompts_module


config_module, paths_module, build_prompts_module = _load_dataset_gen()
PilotDatasetConfig = config_module.PilotDatasetConfig

extract_prompt_text = build_prompts_module.extract_prompt_text
sample_source_rows = build_prompts_module.sample_source_rows
build_domain_prompts = build_prompts_module.build_domain_prompts
assign_splits = build_prompts_module.assign_splits
sanity_check = build_prompts_module.sanity_check


def _fake_source(name: str, n_prompts: int) -> dict:
    return {
        "name": name,
        "hf_repo": f"fake/{name}",
        "hf_subset": None,
        "hf_split": "train",
        "prompt_field": "text",
        "n_prompts": n_prompts,
    }


@pytest.fixture
def tiny_config() -> "PilotDatasetConfig":
    return PilotDatasetConfig(
        in_domain_sources=[_fake_source("fake_a", 10), _fake_source("fake_b", 10)],
        held_out_sources=[_fake_source("fake_c", 5)],
        split_fractions={"train": 0.70, "val": 0.15, "test_indomain": 0.15},
        seed=123,
        output_root="/tmp/build_prompts_test_out",
        work_root="/tmp/build_prompts_test_work",
    )


def _fake_rows(n: int, prefix: str):
    return [(f"{prefix}_row_{i}", f"prompt text {prefix} {i}") for i in range(n)]


def _build_full_prompts_df(config) -> pd.DataFrame:
    """Build a prompts_df for `config` the same way build_prompts_table() does,
    but from synthetic in-memory rows instead of hitting the network."""
    frames = []
    for source in [*config.in_domain_sources, *config.held_out_sources]:
        # Give each source a few more raw rows than n_prompts so sampling is exercised.
        raw_rows = _fake_rows(source["n_prompts"] + 3, source["name"])
        sampled = sample_source_rows(raw_rows, source["n_prompts"], config.seed)
        frames.append(build_domain_prompts(config, source, sampled))
    return pd.concat(frames, ignore_index=True)


# --- extract_prompt_text -------------------------------------------------------

def test_extract_prompt_text_flat_string():
    assert extract_prompt_text("hello world") == "hello world"


def test_extract_prompt_text_conversation_list_user_turn():
    value = [{"role": "user", "content": "what is 2+2?"}]
    assert extract_prompt_text(value) == "what is 2+2?"


def test_extract_prompt_text_conversation_list_prefers_user_over_other_roles():
    value = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the actual question"},
    ]
    assert extract_prompt_text(value) == "the actual question"


def test_extract_prompt_text_conversation_list_falls_back_without_user_role():
    value = [{"role": "assistant", "content": "only an assistant turn"}]
    assert extract_prompt_text(value) == "only an assistant turn"


def test_extract_prompt_text_raises_on_unresolvable_value():
    with pytest.raises(ValueError):
        extract_prompt_text(None)
    with pytest.raises(ValueError):
        extract_prompt_text([{"role": "user", "content": None}])


# --- sample_source_rows ---------------------------------------------------------

def test_sample_source_rows_deterministic_same_seed():
    rows = _fake_rows(20, "x")
    a = sample_source_rows(rows, 5, seed=42)
    b = sample_source_rows(rows, 5, seed=42)
    assert a == b
    assert len(a) == 5


def test_sample_source_rows_different_seed_differs():
    rows = _fake_rows(50, "x")
    a = sample_source_rows(rows, 10, seed=1)
    b = sample_source_rows(rows, 10, seed=2)
    assert a != b


def test_sample_source_rows_clamps_when_fewer_rows_than_requested(capsys):
    rows = _fake_rows(3, "x")
    sampled = sample_source_rows(rows, 10, seed=42)
    assert len(sampled) == 3
    assert set(sampled) == set(rows)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


# --- build_domain_prompts -------------------------------------------------------

def test_build_domain_prompts_fields(tiny_config):
    source = tiny_config.in_domain_sources[0]
    sampled = _fake_rows(3, source["name"])
    df = build_domain_prompts(tiny_config, source, sampled)

    assert list(df.columns) == [
        "prompt_id",
        "domain",
        "source_dataset",
        "source_row_id",
        "prompt_text",
        "held_out_domain",
    ]
    assert list(df["prompt_id"]) == ["fake_a_00000", "fake_a_00001", "fake_a_00002"]
    assert (df["domain"] == "fake_a").all()
    assert (df["source_dataset"] == "fake/fake_a").all()
    assert (df["held_out_domain"] == False).all()  # noqa: E712


def test_build_domain_prompts_held_out_flag(tiny_config):
    source = tiny_config.held_out_sources[0]
    sampled = _fake_rows(2, source["name"])
    df = build_domain_prompts(tiny_config, source, sampled)
    assert (df["held_out_domain"] == True).all()  # noqa: E712


# --- assign_splits + sanity_check (the core invariants) --------------------------

def test_assign_splits_covers_and_partitions_in_domain(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)

    in_domain_ids = set(
        prompts_df.loc[~prompts_df["held_out_domain"], "prompt_id"]
    )
    covered = set(splits["train"]) | set(splits["val"]) | set(splits["test_indomain"])
    assert covered == in_domain_ids

    # no id appears twice across the three in-domain splits
    total_len = len(splits["train"]) + len(splits["val"]) + len(splits["test_indomain"])
    assert total_len == len(in_domain_ids)


def test_assign_splits_heldout_domains_isolated(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)

    held_out_ids = set(prompts_df.loc[prompts_df["held_out_domain"], "prompt_id"])
    assert set(splits["test_heldout_domains"]) == held_out_ids
    assert held_out_ids.isdisjoint(splits["train"])
    assert held_out_ids.isdisjoint(splits["val"])
    assert held_out_ids.isdisjoint(splits["test_indomain"])


def test_sanity_check_passes_on_valid_splits(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)
    # Should not raise.
    sanity_check(tiny_config, prompts_df, splits)


def test_sanity_check_raises_on_cross_split_overlap(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)

    # Inject a duplicate: put a train prompt_id into val too.
    leaked_id = splits["train"][0]
    splits["val"].append(leaked_id)

    with pytest.raises(AssertionError):
        sanity_check(tiny_config, prompts_df, splits)


def test_sanity_check_raises_when_heldout_domain_prompt_leaks_into_train(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)

    held_out_id = splits["test_heldout_domains"][0]
    splits["train"].append(held_out_id)

    with pytest.raises(AssertionError):
        sanity_check(tiny_config, prompts_df, splits)


def test_sanity_check_raises_when_heldout_domain_prompt_missing_from_test_heldout(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)

    # Drop one held-out id entirely from test_heldout_domains without adding it elsewhere.
    splits["test_heldout_domains"] = splits["test_heldout_domains"][1:]

    with pytest.raises(AssertionError):
        sanity_check(tiny_config, prompts_df, splits)


def test_sanity_check_raises_when_in_domain_prompt_leaks_into_test_heldout(tiny_config):
    prompts_df = _build_full_prompts_df(tiny_config)
    splits = assign_splits(tiny_config, prompts_df)

    # Move (not copy) an in-domain id into test_heldout_domains so it isn't
    # also caught by the plain pairwise-overlap check -- this isolates the
    # "test_heldout_domains must contain only held-out ids" invariant.
    stray_id = splits["train"].pop(0)
    splits["test_heldout_domains"].append(stray_id)

    with pytest.raises(AssertionError):
        sanity_check(tiny_config, prompts_df, splits)
