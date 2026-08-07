import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from degeneration_probe.config import DatasetConfig
from degeneration_probe.data.dataset import (
    DegenerationTokenDataset,
    compute_pos_weight,
    derive_bce_targets,
    load_degeneration_records,
    select_rollouts,
)


class FakeTokenizer:
    bos_token = None
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
        return f"PROMPT:{conversation[0]['content']}"

    def __call__(self, text, **kwargs):
        return {"input_ids": [101, 102]}


def test_bce_boundary_is_onset_inclusive():
    assert derive_bce_targets(5, stop_reason="length", onset_position=2) == [0, 0, 1, 1, 1]


def test_eos_is_all_zero_and_undefined_truncation_is_excluded():
    assert derive_bce_targets(3, stop_reason="eos", onset_position=None) == [0, 0, 0]
    assert derive_bce_targets(3, stop_reason="length", onset_position=None) is None


@pytest.fixture
def local_build(tmp_path):
    root = tmp_path / "build"
    (root / "prompts").mkdir(parents=True)
    (root / "onset_labels").mkdir()
    (root / "generations" / "domain_a").mkdir(parents=True)
    (root / "labels" / "domain_a").mkdir(parents=True)

    prompts = pd.DataFrame(
        {
            "prompt_id": ["positive", "negative", "truncated"],
            "prompt_text": ["p+", "p-", "p?"],
        }
    )
    onset = pd.DataFrame(
        {
            "prompt_id": ["positive", "negative", "truncated"],
            "rollout_idx": [0, 0, 0],
            "domain": ["domain_a"] * 3,
            "split": ["train"] * 3,
            "stop_reason": ["length", "eos", "length"],
            "num_tokens": [3, 2, 2],
            "is_positive": [True, False, False],
            "onset_position": [1.0, float("nan"), float("nan")],
        }
    )
    generations = pd.DataFrame(
        {
            "prompt_id": ["positive", "negative", "truncated"],
            "rollout_idx": [0, 0, 0],
            "generated_text": ["a", "b", "c"],
            "generated_token_ids": [[11, 12, 13], [21, 22], [31, 32]],
        }
    )
    labels = pd.DataFrame(
        {
            "prompt_id": ["positive", "negative", "truncated"],
            "rollout_idx": [0, 0, 0],
            "repetition_score": [[0.1, 0.2], [0.3, float("nan")], [0.4, 0.5]],
        }
    )
    prompts.to_parquet(root / "prompts" / "prompts.parquet")
    onset.to_parquet(root / "onset_labels" / "onset_labels.parquet")
    generations.to_parquet(root / "generations" / "domain_a" / "shard_00000.parquet")
    labels.to_parquet(root / "labels" / "domain_a" / "shard_00000.parquet")

    build_yaml = tmp_path / "build.yaml"
    build_data = {
        "model_name": "fake",
        "in_domain_sources": [{"name": "domain_a"}],
        "held_out_sources": [],
    }
    build_yaml.write_text(yaml.safe_dump(build_data))
    (root / "manifest.json").write_text(json.dumps({"config": build_data}))
    config = DatasetConfig(
        short_name="fixture",
        build_root=str(root),
        build_config=str(build_yaml),
        sampling={
            "train_negative_rollouts_per_positive": None,
            "evaluation_negative_rollouts_per_positive": None,
            "domain_stratified": True,
            "seed": 7,
        },
        tokenization={
            "max_length": 8,
            "max_completion_length": 6,
            "prompt_truncation_side": "left",
        },
    )
    return config


def test_local_join_uses_original_ids_and_excludes_undefined_truncation(local_build):
    records = load_degeneration_records(
        local_build, split="train", loss_name="bce", training=True
    )
    assert [record.prompt_id for record in records] == ["negative", "positive"]
    by_id = {record.prompt_id: record for record in records}
    assert by_id["positive"].generated_token_ids.tolist() == [11, 12, 13]
    assert by_id["positive"].targets.tolist() == [0.0, 1.0, 1.0]
    assert by_id["negative"].targets.tolist() == [0.0, 0.0]

    dataset = DegenerationTokenDataset(records, FakeTokenizer(), local_build.tokenization)
    assert compute_pos_weight(dataset) == pytest.approx(3 / 2)


def test_mse_uses_scores_exactly_and_masks_alignment_gap(local_build):
    records = load_degeneration_records(
        local_build, split="train", loss_name="mse", training=True
    )
    positive = next(record for record in records if record.prompt_id == "positive")
    assert positive.targets[:2].tolist() == pytest.approx([0.1, 0.2])
    assert pd.isna(positive.targets[2])
    item = DegenerationTokenDataset(
        [positive], FakeTokenizer(), local_build.tokenization
    )[0]
    assert item["input_ids"].tolist()[-3:] == [11, 12, 13]
    assert item["target_mask"].tolist()[-3:] == [True, True, False]


def test_sampling_is_deterministic_and_excludes_unusable_rows():
    rows = []
    for domain in ("a", "b"):
        rows.append(
            {
                "prompt_id": f"{domain}-positive",
                "rollout_idx": 0,
                "domain": domain,
                "split": "train",
                "stop_reason": "length",
                "is_positive": True,
                "onset_position": 2.0,
            }
        )
        for index in range(5):
            rows.append(
                {
                    "prompt_id": f"{domain}-negative-{index}",
                    "rollout_idx": 0,
                    "domain": domain,
                    "split": "train",
                    "stop_reason": "eos",
                    "is_positive": False,
                    "onset_position": float("nan"),
                }
            )
    rows.append(
        {
            "prompt_id": "unusable",
            "rollout_idx": 0,
            "domain": "a",
            "split": "train",
            "stop_reason": "length",
            "is_positive": False,
            "onset_position": float("nan"),
        }
    )
    frame = pd.DataFrame(rows)
    first = select_rollouts(
        frame,
        split="train",
        negative_rollouts_per_positive=2,
        domain_stratified=True,
        seed=42,
    )
    second = select_rollouts(
        frame,
        split="train",
        negative_rollouts_per_positive=2,
        domain_stratified=True,
        seed=42,
    )
    assert first[["prompt_id", "rollout_idx"]].equals(second[["prompt_id", "rollout_idx"]])
    assert len(first[first["is_positive"]]) == 2
    assert len(first[first["stop_reason"] == "eos"]) == 4
    assert "unusable" not in set(first["prompt_id"])
