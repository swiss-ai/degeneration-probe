"""Tests for degeneration_probe.dataset_gen.cache_activations -- the parts that
don't need a real model/GPU: the activations-manifest resumability logic, and
the safetensors read/write roundtrip with a small fake tensor.

Loads config.py, paths.py, generate.py, and cache_activations.py directly by
file path (bypassing ``degeneration_probe/__init__.py``), the same pattern
used in test_generate.py / test_dataset_gen_paths.py, since the top-level
package ``__init__`` pulls in more than this module needs. cache_activations.py
imports torch and safetensors at module scope (both needed for the pure
tensor/file-format logic under test here) but only imports transformers
lazily inside main(), so loading it here doesn't require network access or a
downloaded model.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GEN_DIR = REPO_ROOT / "degeneration_probe" / "dataset_gen"


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dataset_gen():
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
    generate_module = _load_module(
        "degeneration_probe.dataset_gen.generate", DATASET_GEN_DIR / "generate.py"
    )
    cache_activations_module = _load_module(
        "degeneration_probe.dataset_gen.cache_activations", DATASET_GEN_DIR / "cache_activations.py"
    )
    return config_module, paths_module, generate_module, cache_activations_module


config_module, paths_module, generate_module, cache_activations_module = _load_dataset_gen()
PilotDatasetConfig = config_module.PilotDatasetConfig

ACTIVATIONS_MANIFEST_COLUMNS = cache_activations_module.ACTIVATIONS_MANIFEST_COLUMNS
load_existing_activations_manifest = cache_activations_module.load_existing_activations_manifest
ok_tasks_from_manifest = cache_activations_module.ok_tasks_from_manifest
upsert_manifest_row = cache_activations_module.upsert_manifest_row
write_activations_manifest_atomic = cache_activations_module.write_activations_manifest_atomic
save_activation_file = cache_activations_module.save_activation_file
load_activation_file = cache_activations_module.load_activation_file
build_activation_metadata = cache_activations_module.build_activation_metadata
build_completion_input_ids = cache_activations_module.build_completion_input_ids


# --- manifest: load / ok-filter -------------------------------------------------

def test_load_existing_activations_manifest_missing_file_returns_empty(tmp_path):
    df = load_existing_activations_manifest(tmp_path / "does_not_exist.parquet")
    assert df.empty
    assert list(df.columns) == ACTIVATIONS_MANIFEST_COLUMNS


def _manifest_row(prompt_id, rollout_idx, domain="deepmath_103k", status="ok", shape=None):
    return {
        "prompt_id": prompt_id,
        "rollout_idx": rollout_idx,
        "domain": domain,
        "path": f"/fake/{prompt_id}/rollout_{rollout_idx}.safetensors",
        "shape": shape or [33, 10, 4096],
        "dtype": "float16" if status == "ok" else None,
        "status": status,
        "error_message": None if status == "ok" else "boom",
    }


def test_ok_tasks_from_manifest_only_counts_ok_status():
    df = pd.DataFrame.from_records(
        [
            _manifest_row("p0", 0, status="ok"),
            _manifest_row("p0", 1, status="failed"),
            _manifest_row("p1", 0, status="ok"),
        ],
        columns=ACTIVATIONS_MANIFEST_COLUMNS,
    )
    assert ok_tasks_from_manifest(df) == {("p0", 0), ("p1", 0)}


def test_ok_tasks_from_manifest_empty_df_returns_empty_set():
    df = pd.DataFrame(columns=ACTIVATIONS_MANIFEST_COLUMNS)
    assert ok_tasks_from_manifest(df) == set()


# --- resumability: given a fake existing manifest, confirm already-"ok" pairs
#     are skipped and not recomputed / duplicated ------------------------------

def test_resumability_skips_already_ok_pairs(tmp_path):
    manifest_path = tmp_path / "activations" / "manifest.parquet"
    existing = pd.DataFrame.from_records(
        [
            _manifest_row("deepmath_103k_00000", 0, status="ok"),
            _manifest_row("deepmath_103k_00000", 1, status="failed"),
        ],
        columns=ACTIVATIONS_MANIFEST_COLUMNS,
    )
    write_activations_manifest_atomic(existing, manifest_path)

    reloaded = load_existing_activations_manifest(manifest_path)
    ok_done = ok_tasks_from_manifest(reloaded)
    assert ok_done == {("deepmath_103k_00000", 0)}

    all_tasks = [
        ("deepmath_103k_00000", 0),  # already ok -> should be skipped
        ("deepmath_103k_00000", 1),  # failed -> should be retried
        ("deepmath_103k_00000", 2),  # never seen -> should be processed
    ]
    remaining = [t for t in all_tasks if t not in ok_done]
    assert remaining == [("deepmath_103k_00000", 1), ("deepmath_103k_00000", 2)]


def test_upsert_manifest_row_replaces_same_key_not_duplicates():
    df = pd.DataFrame.from_records(
        [_manifest_row("p0", 0, status="failed")], columns=ACTIVATIONS_MANIFEST_COLUMNS
    )
    updated = upsert_manifest_row(df, _manifest_row("p0", 0, status="ok"))
    assert len(updated) == 1
    assert updated.iloc[0]["status"] == "ok"


def test_upsert_manifest_row_appends_new_key():
    df = pd.DataFrame.from_records(
        [_manifest_row("p0", 0, status="ok")], columns=ACTIVATIONS_MANIFEST_COLUMNS
    )
    updated = upsert_manifest_row(df, _manifest_row("p0", 1, status="ok"))
    assert len(updated) == 2
    assert ok_tasks_from_manifest(updated) == {("p0", 0), ("p0", 1)}


def test_rerun_after_manifest_write_does_not_duplicate_rows(tmp_path):
    """End-to-end: write manifest, 'rerun' processing loop, confirm no duplicate rows."""
    manifest_path = tmp_path / "activations" / "manifest.parquet"

    manifest_df = load_existing_activations_manifest(manifest_path)
    manifest_df = upsert_manifest_row(manifest_df, _manifest_row("p0", 0, status="ok"))
    write_activations_manifest_atomic(manifest_df, manifest_path)

    # "Rerun": load fresh, skip already-ok, process only what's left, write again.
    manifest_df = load_existing_activations_manifest(manifest_path)
    ok_done = ok_tasks_from_manifest(manifest_df)
    all_tasks = [("p0", 0), ("p0", 1)]
    remaining = [t for t in all_tasks if t not in ok_done]
    assert remaining == [("p0", 1)]

    for prompt_id, rollout_idx in remaining:
        manifest_df = upsert_manifest_row(manifest_df, _manifest_row(prompt_id, rollout_idx, status="ok"))
    write_activations_manifest_atomic(manifest_df, manifest_path)

    on_disk = pd.read_parquet(manifest_path)
    assert len(on_disk) == 2
    assert ok_tasks_from_manifest(on_disk) == {("p0", 0), ("p0", 1)}


# --- safetensors read/write roundtrip -------------------------------------------

def test_save_and_load_activation_file_roundtrip(tmp_path):
    path = tmp_path / "deepmath_103k" / "deepmath_103k_00000" / "rollout_0.safetensors"
    fake_hidden_states = torch.randn(33, 10, 4096, dtype=torch.float16)
    metadata = build_activation_metadata("deepmath_103k_00000", 0, completion_len=10)

    save_activation_file(path, fake_hidden_states, metadata)
    assert path.exists()

    loaded_tensor, loaded_metadata = load_activation_file(path)
    assert loaded_tensor.shape == (33, 10, 4096)
    assert loaded_tensor.dtype == torch.float16
    assert torch.equal(loaded_tensor, fake_hidden_states)

    assert loaded_metadata["prompt_id"] == "deepmath_103k_00000"
    assert loaded_metadata["rollout_idx"] == "0"
    assert loaded_metadata["completion_len"] == "10"
    assert loaded_metadata["dtype"] == "float16"
    assert loaded_metadata["layer_order"] == "embedding_then_blocks_0..31"


def test_save_activation_file_does_not_leave_temp_file_on_success(tmp_path):
    path = tmp_path / "rollout_0.safetensors"
    fake_hidden_states = torch.zeros(2, 3, 4, dtype=torch.float16)
    save_activation_file(path, fake_hidden_states, build_activation_metadata("p0", 0, 3))

    leftovers = list(tmp_path.glob(".tmp_act_*"))
    assert leftovers == []


def test_build_activation_metadata_all_values_are_nonempty_strings():
    metadata = build_activation_metadata("p0", 3, completion_len=42)
    assert all(isinstance(v, str) and len(v) > 0 for v in metadata.values())


# --- build_completion_input_ids: alignment with generated_token_ids, no
#     retokenization of the completion (uses a stub tokenizer, no real model) --

class _StubTokenizer:
    """Minimal stand-in: chat-template + tokenization behave predictably."""

    bos_token = "<bos>"

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        assert add_generation_prompt
        user_content = conversation[0]["content"]
        return f"<bos><user>{user_content}<assistant>"

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        assert not add_special_tokens
        # Deterministic fake tokenization: one "token id" per character's ord().
        return {"input_ids": [ord(c) for c in text]}


def test_build_completion_input_ids_appends_generated_ids_verbatim():
    tokenizer = _StubTokenizer()
    generated_token_ids = [999999, 888888, 777777]  # far outside any real char-ord range

    input_ids, prompt_len = build_completion_input_ids(tokenizer, "hi", generated_token_ids)

    # prompt_len must match the retokenized prompt's length exactly.
    expected_prompt_text = "<user>hi<assistant>"  # bos_token stripped, per build_generation_prompt
    assert prompt_len == len(expected_prompt_text)

    # generated_token_ids must appear untouched (not retokenized) at the tail.
    assert input_ids[prompt_len:] == generated_token_ids
    assert input_ids[:prompt_len] == [ord(c) for c in expected_prompt_text]


def test_build_completion_input_ids_completion_len_matches_input():
    tokenizer = _StubTokenizer()
    generated_token_ids = list(range(50))
    input_ids, prompt_len = build_completion_input_ids(tokenizer, "some prompt", generated_token_ids)
    completion_len = len(generated_token_ids)
    assert input_ids[prompt_len : prompt_len + completion_len] == generated_token_ids
    assert len(input_ids) == prompt_len + completion_len
