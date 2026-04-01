"""End-to-end entrypoint for prompt fetch, batch generation, and plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from degeneration_demo.model_utils import resolve_torch_dtype
from degeneration_demo.pipeline import (
    ensure_hf_token,
    fetch_prompt_sample,
    plot_metric_distributions,
    read_jsonl,
    run_prompt_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "outputs" / "prompts_sample.jsonl"
DEFAULT_GENERATIONS_PATH = PROJECT_ROOT / "outputs" / "labeled_generations.jsonl"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch prompts, run batch generation, and plot degeneration metrics."
    )
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Hugging Face dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Optional dataset subset/config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to read from")
    parser.add_argument("--prompt_field", type=str, default=None, help="Explicit field to use as prompt")
    parser.add_argument("--max_samples", type=int, default=25, help="How many prompts to fetch from the dataset")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when shuffling prompts")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="HF model name or local path")
    parser.add_argument("--device", type=str, default="cpu", help="Device map for the model")
    parser.add_argument("--torch_dtype", type=str, default=None, help="torch dtype: auto, float32, float16, bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--chunk_size", type=int, default=256, help="Non-overlapping chunk size for analysis")
    parser.add_argument("--n_values", type=int, nargs="+", default=[1, 3], help="n-gram sizes to analyze")
    parser.add_argument("--max_prompts", type=int, default=10, help="Maximum number of fetched prompts to process")
    parser.add_argument("--token_path", type=str, default=str(PROJECT_ROOT / "keys" / ".hf_token"), help="Path to local HF token file")
    parser.add_argument("--prompts_output", type=str, default=str(DEFAULT_PROMPTS_PATH), help="Where to save fetched prompt sample")
    parser.add_argument("--generations_output", type=str, default=str(DEFAULT_GENERATIONS_PATH), help="Where to save batch generation records")
    parser.add_argument("--plots_output_dir", type=str, default=str(DEFAULT_PLOTS_DIR), help="Where to save histogram plots")
    args = parser.parse_args()

    ensure_hf_token(args.token_path)
    default_dtype = torch.float32 if args.device == "cpu" else None
    torch_dtype = resolve_torch_dtype(args.torch_dtype, default=default_dtype)

    prompts_path = fetch_prompt_sample(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        prompt_field=args.prompt_field,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        output_path=Path(args.prompts_output),
    )

    prompt_records = read_jsonl(prompts_path)[: args.max_prompts]
    generations_path = run_prompt_batch(
        prompt_records=prompt_records,
        model_name=args.model_name,
        device_map=args.device,
        torch_dtype=torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_size=args.chunk_size,
        n_values=args.n_values,
        output_path=Path(args.generations_output),
    )

    generation_records = read_jsonl(generations_path)
    plot_metric_distributions(
        generation_records,
        n_values=args.n_values,
        output_dir=Path(args.plots_output_dir),
    )


if __name__ == "__main__":
    main()
