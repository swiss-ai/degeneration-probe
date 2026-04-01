"""Standalone generation + degeneration-check demo.

Usage examples:
    uv run python scripts/demo_degeneration.py --device cpu
    uv run python scripts/demo_degeneration.py --device cpu --model_name Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import BatchEncoding
from degeneration_demo.model_utils import (
    load_model_and_tokenizer,
    resolve_torch_dtype,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "demo_runs.jsonl"


def is_degenerating_chunk(chunk: List[int], n: int = 1, threshold: float = 0.9) -> Tuple[bool, float]:
    """Return (is_degenerate, repetition_score) for a token-id chunk."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(chunk) < n:
        return False, 0.0

    ngrams = [tuple(chunk[i : i + n]) for i in range(len(chunk) - n + 1)]
    repetition_score = 1.0 - (len(set(ngrams)) / len(ngrams)) if ngrams else 0.0
    return repetition_score > threshold, repetition_score


def ensure_hf_token(token_path: str = str(PROJECT_ROOT / "keys" / ".hf_token")) -> None:
    """If HF token env vars are unset, try loading them from a file."""
    if os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"):
        return

    expanded = os.path.expanduser(token_path)
    try:
        with open(expanded, "r", encoding="utf-8") as f:
            token = f.read().strip()
    except FileNotFoundError:
        token = ""

    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        print(f"Loaded HUGGINGFACE_HUB_TOKEN from {expanded}")


def append_jsonl_record(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_and_check(
    prompt: str,
    model_name: str,
    device_map: str,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_n: int,
    repetition_threshold: float,
    output_path: Path,
) -> None:
    print(f"Loading model {model_name} with device_map={device_map}, dtype={torch_dtype}...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    print("Generating...")
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

    generated_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    is_deg, rep_score = is_degenerating_chunk(
        generated_ids.tolist(), n=repetition_n, threshold=repetition_threshold
    )

    print("\n==== Prompt ====")
    print(prompt)
    print("\n==== Generation ====")
    print(generated_text)
    print("\n==== Degeneration check ====")
    print(f"n-gram size: {repetition_n}")
    print(f"repetition_score: {rep_score:.4f}")
    print(f"degenerating: {is_deg}")

    append_jsonl_record(
        output_path,
        {
            "prompt": prompt,
            "model_name": model_name,
            "generated_text": generated_text,
            "repetition_n": repetition_n,
            "repetition_threshold": repetition_threshold,
            "repetition_score": rep_score,
            "degenerating": is_deg,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    print(f"\nSaved run record to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small instruct model and check for degeneration.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why the sky appears blue during the day in two sentences.",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model name or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device map for the model (cpu, auto, mps, cuda, etc.)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        help="torch dtype: auto, float32, float16, bfloat16",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--repetition_n",
        type=int,
        default=1,
        help="n-gram size for degeneration checker",
    )
    parser.add_argument(
        "--repetition_threshold",
        type=float,
        default=0.9,
        help="Threshold for repetition score to flag degeneration",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="JSONL file where run records will be appended",
    )
    args = parser.parse_args()

    default_dtype = torch.float32 if args.device == "cpu" else None
    torch_dtype = resolve_torch_dtype(args.torch_dtype, default=default_dtype)

    ensure_hf_token(str(PROJECT_ROOT / "keys" / ".hf_token"))

    generate_and_check(
        prompt=args.prompt,
        model_name=args.model_name,
        device_map=args.device,
        torch_dtype=torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_n=args.repetition_n,
        repetition_threshold=args.repetition_threshold,
        output_path=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
