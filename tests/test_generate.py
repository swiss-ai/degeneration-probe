"""Tests for degeneration_probe.dataset_gen.generate -- the parts that don't
need a real model/GPU: seed derivation, the entropy computation given
hand-constructed logits with a known closed-form entropy, and the
resumability logic (skip-already-done + shard/partial consolidation).

Loads config.py, paths.py, and generate.py directly by file path (bypassing
``degeneration_probe/__init__.py``), the same pattern used in
test_dataset_gen_paths.py / test_build_prompts.py, since the top-level package
``__init__`` pulls in more than this module needs. generate.py itself imports
torch (needed for the entropy/sampling tensor ops below) but never imports
transformers at module scope (AutoModelForCausalLM/AutoTokenizer are imported
lazily inside main()), so loading it here doesn't require network access or a
downloaded model.
"""

import importlib.util
import json
import math
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
    return config_module, paths_module, generate_module


config_module, paths_module, generate_module = _load_dataset_gen()
DatasetGenConfig = config_module.DatasetGenConfig

derive_seed = generate_module.derive_seed
compute_entropy_from_logits = generate_module.compute_entropy_from_logits
apply_temperature_and_top_p = generate_module.apply_temperature_and_top_p
all_tasks_for_domain = generate_module.all_tasks_for_domain
compute_remaining_tasks = generate_module.compute_remaining_tasks
load_existing_shard = generate_module.load_existing_shard
done_tasks_from_shard = generate_module.done_tasks_from_shard
write_partial_result = generate_module.write_partial_result
load_partial_results = generate_module.load_partial_results
write_shard_atomic = generate_module.write_shard_atomic
consolidate_shard = generate_module.consolidate_shard
RolloutResult = generate_module.RolloutResult
SHARD_COLUMNS = generate_module.SHARD_COLUMNS


# --- derive_seed ---------------------------------------------------------------

def test_derive_seed_deterministic():
    a = derive_seed(42, "deepmath_103k_00001", 3)
    b = derive_seed(42, "deepmath_103k_00001", 3)
    assert a == b


def test_derive_seed_distinct_across_rollout_idx():
    seeds = {derive_seed(42, "deepmath_103k_00001", r) for r in range(10)}
    assert len(seeds) == 10


def test_derive_seed_distinct_across_prompt_id():
    seeds = {derive_seed(42, f"deepmath_103k_{i:05d}", 0) for i in range(370)}
    assert len(seeds) == 370


def test_derive_seed_distinct_across_config_seed():
    a = derive_seed(42, "deepmath_103k_00001", 0)
    b = derive_seed(43, "deepmath_103k_00001", 0)
    assert a != b


def test_derive_seed_rejects_out_of_range_rollout_idx():
    with pytest.raises(ValueError):
        derive_seed(42, "deepmath_103k_00001", -1)
    with pytest.raises(ValueError):
        derive_seed(42, "deepmath_103k_00001", generate_module.ROLLOUT_IDX_MODULUS)


def test_derive_seed_matches_all_full_scale_prompt_ids_no_collisions():
    """No seed collisions across the full real build's ~3,030 prompt_ids x 10 rollouts.

    Builds synthetic ids matching the real per-domain naming scheme and
    per-domain n_prompts from
    configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml rather
    than depending on the large, environment-specific prompts.parquet file.
    """
    domains = {
        "deepmath_103k": 600,
        "numinamath_1_5": 600,
        "if_sft_data_verified": 600,
        "llama_nemotron": 600,
        "medical_o1": 600,
        "aime_2025": 30,
        "codeforces": 600,
    }
    prompt_ids = [
        f"{domain}_{i:05d}" for domain, n in domains.items() for i in range(n)
    ]
    assert len(prompt_ids) == 3630
    seeds = [derive_seed(42, pid, r) for pid in prompt_ids for r in range(10)]
    assert len(seeds) == len(set(seeds))


# --- compute_entropy_from_logits -----------------------------------------------

def test_entropy_uniform_distribution_matches_log_vocab_size():
    vocab_size = 100
    logits = torch.zeros(vocab_size)  # uniform softmax
    entropy = compute_entropy_from_logits(logits)
    assert entropy.item() == pytest.approx(math.log(vocab_size), abs=1e-5)


def test_entropy_one_hot_like_distribution_is_near_zero():
    vocab_size = 50
    logits = torch.full((vocab_size,), -1e4)
    logits[7] = 1e4  # essentially all mass on one token
    entropy = compute_entropy_from_logits(logits)
    assert entropy.item() == pytest.approx(0.0, abs=1e-5)


def test_entropy_known_closed_form_two_outcomes():
    # logits for a 2-outcome distribution with p = [0.9, 0.1].
    # log(p) - log(1-p) = logit gap; use logits [log(0.9), log(0.1)].
    logits = torch.tensor([math.log(0.9), math.log(0.1)])
    entropy = compute_entropy_from_logits(logits)
    expected = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert entropy.item() == pytest.approx(expected, abs=1e-5)


def test_entropy_batched_matches_per_row_unbatched():
    torch.manual_seed(0)
    batch_logits = torch.randn(4, 37)
    batched = compute_entropy_from_logits(batch_logits)
    for i in range(4):
        single = compute_entropy_from_logits(batch_logits[i])
        assert batched[i].item() == pytest.approx(single.item(), abs=1e-6)


def test_entropy_lower_for_predictable_than_open_ended_distribution():
    # Sanity check mirroring the dry-run spot check: a near-deterministic
    # ("predictable continuation") distribution should have much lower
    # entropy than a near-uniform ("open-ended") one over the same vocab.
    vocab_size = 200
    predictable_logits = torch.full((vocab_size,), -1e4)
    predictable_logits[3] = 1e4
    open_ended_logits = torch.randn(vocab_size) * 0.1  # close to uniform

    predictable_entropy = compute_entropy_from_logits(predictable_logits).item()
    open_ended_entropy = compute_entropy_from_logits(open_ended_logits).item()
    assert predictable_entropy < open_ended_entropy


# --- apply_temperature_and_top_p -----------------------------------------------

def test_top_p_filtering_keeps_only_nucleus_tokens():
    # Four tokens with clean probabilities under temperature=1: [0.75, 0.2, 0.04, 0.01].
    logits = torch.log(torch.tensor([0.75, 0.2, 0.04, 0.01]))
    filtered = apply_temperature_and_top_p(logits, temperature=1.0, top_p=0.9)
    kept = torch.isfinite(filtered)
    # cumulative: 0.75, 0.95, 0.99, 1.0 -> first two tokens' cumulative already
    # exceeds 0.9, so only those two survive.
    assert kept.tolist() == [True, True, False, False]


def test_top_p_always_keeps_top_token_even_if_alone_exceeds_top_p():
    logits = torch.log(torch.tensor([0.95, 0.03, 0.02]))
    filtered = apply_temperature_and_top_p(logits, temperature=1.0, top_p=0.5)
    kept = torch.isfinite(filtered)
    assert kept.tolist() == [True, False, False]


def test_top_p_one_disables_filtering():
    logits = torch.randn(50)
    filtered = apply_temperature_and_top_p(logits, temperature=1.0, top_p=1.0)
    assert torch.isfinite(filtered).all()


def test_temperature_scaling_applied_before_filtering():
    logits = torch.tensor([2.0, 1.0, 0.0])
    filtered_t1 = apply_temperature_and_top_p(logits, temperature=1.0, top_p=1.0)
    filtered_t2 = apply_temperature_and_top_p(logits, temperature=2.0, top_p=1.0)
    assert torch.allclose(filtered_t2, logits / 2.0)
    assert torch.allclose(filtered_t1, logits / 1.0)


# --- resumability: all_tasks_for_domain / compute_remaining_tasks -------------

def _prompts_df(domain_counts):
    records = []
    for domain, n in domain_counts.items():
        for i in range(n):
            records.append({"prompt_id": f"{domain}_{i:05d}", "domain": domain, "prompt_text": "x"})
    return pd.DataFrame.from_records(records)


def test_all_tasks_for_domain_cross_product():
    df = _prompts_df({"deepmath_103k": 3, "aime_2025": 2})
    tasks = all_tasks_for_domain(df, "deepmath_103k", n_rollouts=4)
    assert len(tasks) == 12
    assert set(tasks) == {
        (f"deepmath_103k_{i:05d}", r) for i in range(3) for r in range(4)
    }


def test_compute_remaining_tasks_skips_done():
    all_tasks = [("p0", 0), ("p0", 1), ("p1", 0), ("p1", 1)]
    done = {("p0", 0), ("p1", 1)}
    remaining = compute_remaining_tasks(all_tasks, done)
    assert remaining == [("p0", 1), ("p1", 0)]


def test_compute_remaining_tasks_all_done_is_empty():
    all_tasks = [("p0", 0), ("p0", 1)]
    done = {("p0", 0), ("p0", 1)}
    assert compute_remaining_tasks(all_tasks, done) == []


# --- resumability: shard / partial-result persistence -------------------------

def _fake_result(prompt_id, rollout_idx, seed=123) -> "RolloutResult":
    return RolloutResult(
        prompt_id=prompt_id,
        rollout_idx=rollout_idx,
        seed=seed,
        generated_token_ids=[1, 2, 3],
        generated_text="abc",
        per_token_entropy=[0.1, 0.2, 0.3],
        num_tokens=3,
        stop_reason="eos",
        wall_time_seconds=0.5,
    )


def test_load_existing_shard_missing_file_returns_empty(tmp_path):
    df = load_existing_shard(tmp_path / "does_not_exist.parquet")
    assert df.empty
    assert list(df.columns) == SHARD_COLUMNS


def test_done_tasks_from_shard(tmp_path):
    shard_path = tmp_path / "shard_00000.parquet"
    df = pd.DataFrame.from_records(
        [asdict_result(_fake_result("p0", 0)), asdict_result(_fake_result("p0", 1))],
        columns=SHARD_COLUMNS,
    )
    df.to_parquet(shard_path, index=False)
    loaded = load_existing_shard(shard_path)
    assert done_tasks_from_shard(loaded) == {("p0", 0), ("p0", 1)}


def asdict_result(result):
    from dataclasses import asdict
    return asdict(result)


def test_write_and_load_partial_result_roundtrip(tmp_path):
    work_dir = tmp_path / "work"
    result = _fake_result("p0", 2, seed=999)
    path = write_partial_result(work_dir, result)
    assert path.exists()

    loaded = load_partial_results(work_dir)
    assert ("p0", 2) in loaded
    assert loaded[("p0", 2)]["seed"] == 999
    assert loaded[("p0", 2)]["generated_token_ids"] == [1, 2, 3]


def test_load_partial_results_ignores_leftover_tmp_files(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    # Simulate a crash mid-write: a stray temp file that never got renamed.
    (work_dir / ".tmp_partial_abc123.json").write_text("{not valid json")
    write_partial_result(work_dir, _fake_result("p0", 0))

    loaded = load_partial_results(work_dir)
    assert set(loaded.keys()) == {("p0", 0)}


def test_load_partial_results_skips_corrupted_json(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "corrupted__0000.json").write_text("{not valid json")
    write_partial_result(work_dir, _fake_result("p0", 0))

    loaded = load_partial_results(work_dir)
    assert set(loaded.keys()) == {("p0", 0)}


def test_consolidate_shard_merges_partials_into_new_shard(tmp_path):
    shard_path = tmp_path / "generations" / "deepmath_103k" / "shard_00000.parquet"
    work_dir = tmp_path / "work" / "deepmath_103k"

    write_partial_result(work_dir, _fake_result("p0", 0))
    write_partial_result(work_dir, _fake_result("p0", 1))

    combined = consolidate_shard(shard_path, work_dir)
    assert done_tasks_from_shard(combined) == {("p0", 0), ("p0", 1)}
    assert shard_path.exists()

    on_disk = pd.read_parquet(shard_path)
    assert done_tasks_from_shard(on_disk) == {("p0", 0), ("p0", 1)}


def test_consolidate_shard_is_idempotent_and_resumable(tmp_path):
    """End-to-end resumability check: partial run -> simulated interrupt -> rerun.

    Round 1 "generates" p0/rollout 0 and writes it; consolidate. Round 2
    (simulating a fresh process after a crash/interrupt) sees the existing
    shard, "generates" only the remaining task (p0/rollout 1, using
    compute_remaining_tasks to find it), and consolidates again. No rows are
    duplicated or lost.
    """
    shard_path = tmp_path / "generations" / "deepmath_103k" / "shard_00000.parquet"
    work_dir = tmp_path / "work" / "deepmath_103k"
    all_tasks = [("p0", 0), ("p0", 1)]

    # --- round 1: only rollout 0 completes before the "interrupt" ---
    write_partial_result(work_dir, _fake_result("p0", 0))
    consolidate_shard(shard_path, work_dir)

    done_after_round_1 = done_tasks_from_shard(load_existing_shard(shard_path))
    assert done_after_round_1 == {("p0", 0)}
    remaining_after_round_1 = compute_remaining_tasks(all_tasks, done_after_round_1)
    assert remaining_after_round_1 == [("p0", 1)]

    # --- round 2: fresh "process" resumes, generates only what's remaining ---
    existing_df = load_existing_shard(shard_path)
    partials_seen = load_partial_results(work_dir)
    done_before_round_2 = done_tasks_from_shard(existing_df) | set(partials_seen.keys())
    remaining_before_round_2 = compute_remaining_tasks(all_tasks, done_before_round_2)
    assert remaining_before_round_2 == [("p0", 1)]

    write_partial_result(work_dir, _fake_result("p0", 1))
    final = consolidate_shard(shard_path, work_dir)

    # Exactly one row per task -- no duplicates, nothing lost.
    on_disk = pd.read_parquet(shard_path)
    assert len(on_disk) == 2
    assert done_tasks_from_shard(on_disk) == {("p0", 0), ("p0", 1)}
    assert done_tasks_from_shard(final) == {("p0", 0), ("p0", 1)}


def test_consolidate_shard_no_partials_returns_existing_unchanged(tmp_path):
    shard_path = tmp_path / "generations" / "deepmath_103k" / "shard_00000.parquet"
    work_dir = tmp_path / "work" / "deepmath_103k"

    write_partial_result(work_dir, _fake_result("p0", 0))
    consolidate_shard(shard_path, work_dir)
    mtime_before = shard_path.stat().st_mtime_ns

    # No new partials since -- consolidating again should not rewrite the file.
    consolidate_shard(shard_path, work_dir)
    mtime_after = shard_path.stat().st_mtime_ns
    assert mtime_before == mtime_after
