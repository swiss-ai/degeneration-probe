"""Fetch prompts, generate responses, compute TTR (n=1,2,3), and label with OpenAI.

Example:
    uv run python scripts/generate_and_label.py \
      --max_samples 200 \
      --device cpu \
      --model_name Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os

import torch
from openai import OpenAI

from degeneration_probe.metrics import summarize_chunks
from degeneration_probe.model_utils import load_model_and_tokenizer, resolve_torch_dtype
from degeneration_probe.data import (
    append_jsonl,
    ensure_hf_token,
    fetch_prompt_sample,
    read_jsonl,
)
from degeneration_probe.generation import generate_for_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_VALUES = [1, 2, 3]

LABELING_PROMPT = """\
You are evaluating whether a language model's output shows signs of text degeneration.

Degeneration means the model got stuck in a repetitive loop, producing the same \
words, phrases, or patterns over and over. This is different from:
- Naturally listing items (not degeneration)
- Using some repeated words in normal prose (not degeneration)
- Repetition for rhetorical emphasis (not degeneration)

Only label as degenerate if the output clearly shows the model is stuck in an \
unintentional repetitive loop.

Prompt given to the model:
{prompt}

Model's response:
{response}

Is this response degenerate? Answer with exactly one word: "yes" or "no"."""


def label_with_openai(
    client: OpenAI, prompt: str, response: str, model: str = "gpt-4o-mini"
) -> bool:
    """Ask OpenAI whether a generation is degenerate."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": LABELING_PROMPT.format(prompt=prompt, response=response),
            }
        ],
        max_tokens=5,
        temperature=0.0,
    )
    answer = completion.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate responses and label degeneration with OpenAI."
    )
    # Dataset args
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_field", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    # Generation args
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--chunk_size", type=int, default=256)
    # Labeling args
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "threshold_data.jsonl"),
    )
    parser.add_argument(
        "--token_path",
        type=str,
        default=str(PROJECT_ROOT / "keys" / ".hf_token"),
    )
    args = parser.parse_args()

    ensure_hf_token(args.token_path)

    # ---- 1. Fetch prompts ----
    prompts_path = PROJECT_ROOT / "outputs" / "threshold_prompts.jsonl"
    fetch_prompt_sample(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        prompt_field=args.prompt_field,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        output_path=prompts_path,
    )
    prompt_records = read_jsonl(prompts_path)

    # ---- 2. Load generation model ----
    default_dtype = torch.float32 if args.device == "cpu" else None
    torch_dtype = resolve_torch_dtype(args.torch_dtype, default=default_dtype)
    print(f"Loading model {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch_dtype,
    )

    # ---- 3. OpenAI client ----
    openai_key_path = PROJECT_ROOT / "keys" / ".openai_key"
    api_key = openai_key_path.read_text(encoding="utf-8").strip()
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)

    # ---- 4. Generate, measure, label ----
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for idx, record in enumerate(prompt_records, start=1):
        prompt = record["prompt"]
        print(f"[{idx}/{len(prompt_records)}] {record['prompt_id']}")

        generated_ids, generated_text = generate_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        chunk_summary = summarize_chunks(
            generated_ids, n_values=N_VALUES, chunk_size=args.chunk_size
        )

        llm_label = label_with_openai(
            client, prompt, generated_text, model=args.openai_model
        )
        print(f"  tokens={len(generated_ids)}  degenerating={llm_label}")

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
                "llm_label": llm_label,
            },
        )

    print(f"\nDone. Saved {len(prompt_records)} records to {output_path}")


if __name__ == "__main__":
    main()
