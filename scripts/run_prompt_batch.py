"""Generate on many prompts and save chunk-based degeneration metrics.

Example:
    uv run python scripts/run_prompt_batch.py \
      --input_path outputs/prompts_sample.jsonl \
      --output_path outputs/labeled_generations.jsonl \
      --max_prompts 10 \
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import BatchEncoding

from degeneration_demo.metrics import summarize_chunks
from degeneration_demo.model_utils import load_model_and_tokenizer, resolve_torch_dtype

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "outputs" / "prompts_sample.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "labeled_generations.jsonl"


def ensure_hf_token(token_path: str = str(PROJECT_ROOT / "keys" / ".hf_token")) -> None:
    """If HF token env vars are unset, try loading them from a file."""
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


def load_prompt_records(path: Path) -> list[dict]:
    """Read prompt records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_for_prompt(
    model,
    tokenizer,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation on many prompts and save chunk metrics.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH), help="Prompt JSONL input file")
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Generation JSONL output file")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="HF model name or local path")
    parser.add_argument("--device", type=str, default="cpu", help="Device map for the model")
    parser.add_argument("--torch_dtype", type=str, default=None, help="torch dtype: auto, float32, float16, bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--chunk_size", type=int, default=256, help="Non-overlapping chunk size for analysis")
    parser.add_argument("--n_values", type=int, nargs="+", default=[1, 3], help="n-gram sizes to analyze")
    parser.add_argument("--max_prompts", type=int, default=10, help="Maximum number of prompts to process")
    args = parser.parse_args()

    ensure_hf_token()
    default_dtype = torch.float32 if args.device == "cpu" else None
    torch_dtype = resolve_torch_dtype(args.torch_dtype, default=default_dtype)

    prompt_records = load_prompt_records(Path(args.input_path))
    prompt_records = prompt_records[: args.max_prompts]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    print(f"Loading model {args.model_name} with device_map={args.device}, dtype={torch_dtype}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch_dtype,
    )

    for idx, record in enumerate(prompt_records, start=1):
        prompt = record["prompt"]
        print(f"[{idx}/{len(prompt_records)}] Generating for prompt_id={record['prompt_id']}")
        generated_ids, generated_text = generate_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        chunk_summary = summarize_chunks(generated_ids, n_values=args.n_values, chunk_size=args.chunk_size)

        binary_label = chunk_summary["metrics_by_n"][str(args.n_values[0])]["max_repetition"] > 0.9
        append_jsonl(
            output_path,
            {
                "prompt_id": record["prompt_id"],
                "prompt": prompt,
                "source_dataset": record.get("source_dataset"),
                "model_name": args.model_name,
                "generated_text": generated_text,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "chunk_summary": chunk_summary,
                "degenerating": binary_label,
            },
        )

    print(f"Done. Saved {len(prompt_records)} records to {output_path}")


if __name__ == "__main__":
    main()
