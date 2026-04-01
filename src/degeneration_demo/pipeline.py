"""Reusable helpers for the degeneration demo pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import BatchEncoding

from degeneration_demo.metrics import summarize_chunks
from degeneration_demo.model_utils import load_model_and_tokenizer


def ensure_hf_token(token_path: str) -> None:
    """If HF token env vars are unset, try loading them from a local file."""
    if os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"):
        return

    try:
        token = Path(token_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""

    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        print(f"Loaded HUGGINGFACE_HUB_TOKEN from {token_path}")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
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

    raise ValueError("Could not infer a prompt field. Pass --prompt_field explicitly for this dataset.")


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
    """Fetch prompts from HF and save a local JSONL sample."""
    print(f"Loading dataset: {dataset_name} (subset={subset}, split={split})")
    dataset = load_dataset(dataset_name, subset, split=split)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    limit = min(max_samples, len(dataset))
    print(f"Saving {limit} prompts to {output_path}")
    for idx in range(limit):
        example = dataset[idx]
        prompt = build_prompt_from_example(example, prompt_field=prompt_field)
        append_jsonl(
            output_path,
            {
                "prompt_id": f"{dataset_name.replace('/', '_')}-{split}-{idx}",
                "prompt": prompt,
                "source_dataset": dataset_name,
            },
        )

    print(f"Done. Saved {limit} prompts.")
    return output_path


def generate_for_prompt(
    model,
    tokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[int], str]:
    """Generate token ids and text for one prompt."""
    messages = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )

    device = getattr(model, "device", torch.device("cpu"))
    if isinstance(model_inputs, BatchEncoding):
        model_inputs = model_inputs.to(device)
        input_ids = model_inputs["input_ids"]
    else:
        input_ids = model_inputs.to(device)
        model_inputs = {"input_ids": input_ids}

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, generated_text


def run_prompt_batch(
    *,
    prompt_records: list[dict[str, Any]],
    model_name: str,
    device_map: str,
    torch_dtype,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    chunk_size: int,
    n_values: list[int],
    output_path: Path,
) -> Path:
    """Generate on many prompts and save chunk-level metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    print(f"Loading model {model_name} with device_map={device_map}, dtype={torch_dtype}...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    for idx, record in enumerate(prompt_records, start=1):
        prompt = record["prompt"]
        print(f"[{idx}/{len(prompt_records)}] Generating for prompt_id={record['prompt_id']}")
        generated_ids, generated_text = generate_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        chunk_summary = summarize_chunks(generated_ids, n_values=n_values, chunk_size=chunk_size)
        default_n = str(n_values[0])
        binary_label = chunk_summary["metrics_by_n"][default_n]["max_repetition"] > 0.9
        append_jsonl(
            output_path,
            {
                "prompt_id": record["prompt_id"],
                "prompt": prompt,
                "source_dataset": record.get("source_dataset"),
                "model_name": model_name,
                "generated_text": generated_text,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "chunk_summary": chunk_summary,
                "degenerating": binary_label,
            },
        )

    print(f"Done. Saved {len(prompt_records)} records to {output_path}")
    return output_path


def collect_metric(records: list[dict[str, Any]], *, n: int, metric_name: str) -> list[float]:
    """Collect one aggregated metric across all generation records."""
    values = []
    for record in records:
        metrics_by_n = record.get("chunk_summary", {}).get("metrics_by_n", {})
        metric_block = metrics_by_n.get(str(n))
        if metric_block and metric_name in metric_block:
            values.append(metric_block[metric_name])
    return values


def save_histogram(values: list[float], *, title: str, xlabel: str, output_path: Path) -> None:
    """Save a histogram to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=20, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metric_distributions(records: list[dict[str, Any]], *, n_values: list[int], output_dir: Path) -> Path:
    """Plot histogram distributions for chunk TTR and repetition metrics."""
    for n in n_values:
        min_ttr_values = collect_metric(records, n=n, metric_name="min_ttr")
        max_rep_values = collect_metric(records, n=n, metric_name="max_repetition")

        if min_ttr_values:
            save_histogram(
                min_ttr_values,
                title=f"Distribution of min chunk TTR for {n}-grams",
                xlabel="min chunk TTR",
                output_path=output_dir / f"ttr_n{n}.png",
            )
        if max_rep_values:
            save_histogram(
                max_rep_values,
                title=f"Distribution of max chunk repetition for {n}-grams",
                xlabel="max chunk repetition",
                output_path=output_dir / f"repetition_n{n}.png",
            )

    print(f"Saved plots to {output_dir}")
    return output_dir
