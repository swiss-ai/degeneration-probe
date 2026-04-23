"""LLM generation and batch processing with degeneration metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import torch
from transformers import BatchEncoding

from degeneration_probe.data import append_jsonl
from degeneration_probe.metrics import summarize_chunks
from degeneration_probe.model_utils import load_model_and_tokenizer


def generate_for_prompt(
    model,
    tokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[int], str]:
    """Generate token ids and decoded text for one prompt."""
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
    else:
        # Base models without a chat template: tokenize the raw prompt.
        model_inputs = tokenizer(prompt, return_tensors="pt")

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
    prompt_records: List[dict[str, Any]],
    model_name: str,
    device_map: str,
    torch_dtype: Optional[torch.dtype],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    chunk_size: int,
    n_values: List[int],
    output_path: Path,
) -> Path:
    """Generate completions for a batch of prompts and save chunk-level metrics.

    Each record written to ``output_path`` includes the prompt, generated text,
    chunk-based degeneration metrics, and a binary ``degenerating`` label.
    """
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
