"""Fetch a small prompt sample from a Hugging Face dataset and save it locally.

Example:
    uv run python scripts/fetch_prompts.py \
      --dataset tatsu-lab/alpaca \
      --split train \
      --max_samples 25
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "prompts_sample.jsonl"


def build_prompt_from_example(example: dict[str, Any], prompt_field: Optional[str] = None) -> str:
    """Extract or build a prompt string from a dataset row."""
    if prompt_field:
        value = example.get(prompt_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{prompt_field}' is missing or empty in example: {example}")
        return value.strip()

    # Common prompt-like fields
    for candidate in ("prompt", "question", "instruction", "query"):
        value = example.get(candidate)
        if isinstance(value, str) and value.strip():
            if candidate == "instruction" and isinstance(example.get("input"), str) and example["input"].strip():
                return f"Instruction: {value.strip()}\nInput: {example['input'].strip()}"
            return value.strip()

    # Chat-style datasets
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

    raise ValueError(
        "Could not infer a prompt field. Pass --prompt_field explicitly for this dataset."
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch prompt samples from a HF dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset/config name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to read from",
    )
    parser.add_argument(
        "--prompt_field",
        type=str,
        default=None,
        help="Explicit field to use as prompt if auto-detection fails",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=25,
        help="How many prompts to save",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the JSONL sample",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} (subset={args.subset}, split={args.split})")
    dataset = load_dataset(args.dataset, args.subset, split=args.split)

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    limit = min(args.max_samples, len(dataset))
    print(f"Saving {limit} prompts to {output_path}")

    saved = 0
    for idx in range(limit):
        example = dataset[idx]
        prompt = build_prompt_from_example(example, prompt_field=args.prompt_field)
        append_jsonl(
            output_path,
            {
                "prompt_id": f"{args.dataset.replace('/', '_')}-{args.split}-{idx}",
                "prompt": prompt,
                "source_dataset": args.dataset,
            },
        )
        saved += 1

    print(f"Done. Saved {saved} prompts.")


if __name__ == "__main__":
    main()
