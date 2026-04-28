"""Prompt fetching and JSONL I/O utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Prompt cache helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_cache_dir() -> Path:
    """Return the prompt cache directory.

    Controlled by the ``PROBE_DATA_CACHE`` env var.  Falls back to
    ``data/cache/`` relative to the project root so it is shared across runs
    but excluded from git.
    """
    env = os.environ.get("PROBE_DATA_CACHE")
    return Path(env) if env else _PROJECT_ROOT / "data" / "cache"


def _prompt_cache_path(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    seed: int,
    max_samples: int,
    shuffle: bool,
) -> Path:
    """Build a deterministic cache file path for a given prompt sample config."""
    slug = dataset_name.replace("/", "__")
    subset_part = subset if subset else "none"
    shuffle_part = "_shuffled" if shuffle else ""
    filename = f"{slug}_{subset_part}_{split}_s{seed}_n{max_samples}{shuffle_part}.jsonl"
    return _resolve_cache_dir() / filename


# ---------------------------------------------------------------------------


def ensure_hf_token() -> None:
    """Mirror ``HF_TOKEN`` into ``HUGGINGFACE_HUB_TOKEN`` (and vice versa) so
    both forms work regardless of which one the user / sbatch script set.

    The canonical place to define the token is the project-root ``.env`` file
    (loaded by ``cluster/clariden_*.sh`` automatically; locally, source it
    yourself or add it to your shell rc).
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: List[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt_from_example(example: dict[str, Any], prompt_field: Optional[str] = None) -> str:
    """Extract or build a prompt string from a dataset row."""
    if prompt_field:
        value = example.get(prompt_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{prompt_field}' is missing or empty in example: {example}")
        return value.strip()

    for candidate in ("prompt", "question", "instruction", "query"):
        value = example.get(candidate)
        if isinstance(value, str) and value.strip():
            if candidate == "instruction" and isinstance(example.get("input"), str) and example["input"].strip():
                return f"Instruction: {value.strip()}\nInput: {example['input'].strip()}"
            return value.strip()

    # Standard messages format: [{"role": "user", "content": "..."}]
    messages = example.get("messages")
    if isinstance(messages, list):
        user_messages = []
        for item in messages:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    user_messages.append(content.strip())
        if user_messages:
            return "\n".join(user_messages)

    # ShareGPT / OpenHermes format: [{"from": "human", "value": "..."}]
    conversations = example.get("conversations")
    if isinstance(conversations, list):
        user_messages = []
        for item in conversations:
            if isinstance(item, dict) and item.get("from") == "human":
                content = item.get("value")
                if isinstance(content, str) and content.strip():
                    user_messages.append(content.strip())
        if user_messages:
            return "\n".join(user_messages)

    raise ValueError("Could not infer a prompt field. Set 'prompt_field' explicitly in your config.")


def resolve_model_from_data(paths: list[str | Path]) -> str:
    """Read the model_name from JSONL data files and validate consistency.

    Returns the shared model name. Raises ValueError if files disagree or
    if model_name is missing from any file.
    """
    model_names: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        with open(p, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Data file is empty: {p}")
        record = json.loads(first_line)
        name = record.get("model_name")
        if not name:
            raise ValueError(
                f"Data file {p} is missing 'model_name' field. "
                "Was it generated with an older pipeline version?"
            )
        model_names[str(p)] = name

    unique = set(model_names.values())
    if len(unique) > 1:
        details = "\n".join(f"  {path}: {name}" for path, name in model_names.items())
        raise ValueError(
            f"Model mismatch across training data files:\n{details}\n"
            "All training data must come from the same model."
        )

    return unique.pop()


def fetch_prompt_sample(
    *,
    dataset_name: str,
    subset: Optional[str],
    split: str,
    prompt_field: Optional[str],
    max_samples: int,
    shuffle: bool,
    seed: int,
    output_path: Path,
) -> Path:
    """Fetch prompts from a HuggingFace dataset and save a local JSONL sample.

    On subsequent calls with identical parameters the prompts are loaded from a
    local cache (``data/cache/`` by default, or ``$PROBE_DATA_CACHE``) instead
    of re-downloading from HuggingFace Hub.
    """
    cache_path = _prompt_cache_path(dataset_name, subset, split, seed, max_samples, shuffle)

    if cache_path.exists():
        records = read_jsonl(cache_path)
        print(f"[cache] Loaded {len(records)} prompts from {cache_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        for record in records:
            append_jsonl(output_path, record)
        return output_path

    print(f"Loading dataset: {dataset_name} (subset={subset}, split={split})")
    dataset = load_dataset(dataset_name, subset, split=split)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    limit = min(max_samples, len(dataset))
    print(f"Saving {limit} prompts to {output_path} (caching at {cache_path})")
    for idx in range(limit):
        example = dataset[idx]
        prompt = build_prompt_from_example(example, prompt_field=prompt_field)
        record = {
            "prompt_id": f"{dataset_name.replace('/', '_')}-{split}-{idx}",
            "prompt": prompt,
            "source_dataset": dataset_name,
        }
        append_jsonl(output_path, record)
        append_jsonl(cache_path, record)

    print(f"Done. Saved {limit} prompts.")
    return output_path
