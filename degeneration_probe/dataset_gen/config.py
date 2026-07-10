"""Configuration for the dataset generation pipeline.

This module only defines the configuration schema (``DatasetGenConfig``) plus
YAML round-tripping helpers. It intentionally contains no generation, labeling,
or activation-extraction logic -- see ``degeneration_probe/dataset_gen/paths.py``
and ``degeneration_probe/dataset_gen/manifest.py`` for the other scaffolding
pieces, and other modules (added separately) for the actual pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

# Required keys for every entry in `in_domain_sources` / `held_out_sources`.
SOURCE_REQUIRED_KEYS = {"name", "hf_repo", "hf_subset", "hf_split", "prompt_field", "n_prompts"}

# Default storage locations for the dataset. These point at this pipeline's
# author's own capstor paths -- anyone building their own copy of the dataset
# (rather than reading the shared one) should override output_root/work_root
# to a path they can write to. See configs/dataset/full_scale.yaml and
# notebooks/inspect_dataset.ipynb Section 1 for the canonical build's actual
# location and how to point a fresh build elsewhere.
DEFAULT_OUTPUT_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/dataset_full_scale"
)
DEFAULT_WORK_ROOT = Path(
    "/capstor/scratch/cscs/mdenegri/degeneration-probe/dataset_full_scale_work"
)


def _default_in_domain_sources() -> List[Dict[str, Any]]:
    return [
        {
            "name": "deepmath_103k",
            "hf_repo": "zwhe99/DeepMath-103K",
            "hf_subset": None,
            "hf_split": "train",
            "prompt_field": "question",
            "n_prompts": 600,
        },
        {
            "name": "numinamath_1_5",
            "hf_repo": "AI-MO/NuminaMath-1.5",
            "hf_subset": None,
            "hf_split": "train",
            "prompt_field": "problem",
            "n_prompts": 600,
        },
        {
            "name": "if_sft_data_verified",
            "hf_repo": "allenai/IF_sft_data_verified",
            "hf_subset": None,
            "hf_split": "train",
            "prompt_field": "messages",
            "n_prompts": 600,
        },
        {
            "name": "llama_nemotron",
            "hf_repo": "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "hf_subset": "SFT",
            "hf_split": "code",
            "prompt_field": "input",
            "n_prompts": 600,
        },
    ]


def _default_held_out_sources() -> List[Dict[str, Any]]:
    return [
        {
            "name": "medical_o1",
            "hf_repo": "FreedomIntelligence/medical-o1-verifiable-problem",
            "hf_subset": None,
            "hf_split": "train",
            "prompt_field": "Open-ended Verifiable Question",
            "n_prompts": 600,
        },
        {
            # MathArena/aime_2025's "train" split only has 30 rows total
            # (AIME I + II 2025, 15 problems each) -- n_prompts is already
            # capped at that real maximum; sample_source_rows() clamps to
            # the available row count and warns rather than failing.
            "name": "aime_2025",
            "hf_repo": "MathArena/aime_2025",
            "hf_subset": None,
            "hf_split": "train",
            "prompt_field": "problem",
            "n_prompts": 30,
        },
    ]


def _default_split_fractions() -> Dict[str, float]:
    return {"train": 0.70, "val": 0.15, "test_indomain": 0.15}


@dataclass
class DatasetGenConfig:
    """Configuration for generating the dataset.

    ``in_domain_sources`` and ``held_out_sources`` are lists of dicts, each
    describing one HF dataset to draw prompts from:
        name: short identifier used in paths/manifests (e.g. "deepmath_103k")
        hf_repo: HF dataset repo id (e.g. "zwhe99/DeepMath-103K")
        hf_subset: HF dataset config/subset name, or None
        hf_split: HF dataset split to read from (e.g. "train")
        prompt_field: name of the column holding the raw prompt text
        n_prompts: number of prompts to sample from this source

    The field defaults below mirror configs/dataset/full_scale.yaml (the
    current build); load that file directly via ``from_yaml`` rather than
    relying on these defaults for anything other than quick/ad hoc use.
    """

    model_name: str = "swiss-ai/Apertus-8B-Instruct-2509"

    in_domain_sources: List[Dict[str, Any]] = field(default_factory=_default_in_domain_sources)
    held_out_sources: List[Dict[str, Any]] = field(default_factory=_default_held_out_sources)

    n_rollouts_per_prompt: int = 10
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42

    # Storage roots.
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT
    work_root: Union[str, Path] = DEFAULT_WORK_ROOT

    # How the in-domain prompt pool is split. Held-out sources are not split
    # further -- they are entirely reserved for out-of-distribution eval.
    split_fractions: Dict[str, float] = field(default_factory=_default_split_fractions)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.work_root = Path(self.work_root)

        for group_name in ("in_domain_sources", "held_out_sources"):
            sources = getattr(self, group_name)
            for source in sources:
                missing = SOURCE_REQUIRED_KEYS - source.keys()
                if missing:
                    raise ValueError(
                        f"{group_name} entry {source!r} is missing required keys: "
                        f"{sorted(missing)}"
                    )

        fractions_total = sum(self.split_fractions.values())
        if abs(fractions_total - 1.0) > 1e-6:
            raise ValueError(
                f"split_fractions must sum to 1.0, got {fractions_total} "
                f"({self.split_fractions!r})"
            )

        if self.n_rollouts_per_prompt <= 0:
            raise ValueError("n_rollouts_per_prompt must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain (JSON/YAML-friendly) dict, e.g. for the manifest."""
        raw = asdict(self)
        raw["output_root"] = str(self.output_root)
        raw["work_root"] = str(self.work_root)
        return raw

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetGenConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DatasetGenConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
