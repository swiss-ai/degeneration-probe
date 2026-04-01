"""Entry point: fetch prompts, run LLM, compute degeneration metrics, save plots."""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path

from degeneration_probe.config import load_config
from degeneration_probe.data import ensure_hf_token, fetch_prompt_sample, read_jsonl
from degeneration_probe.generation import run_prompt_batch
from degeneration_probe.model_utils import resolve_torch_dtype
from degeneration_probe.plotting import plot_metric_distributions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the degeneration-probe data collection pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML experiment config (e.g. configs/alpaca_qwen05b.yaml)",
    )
    parser.add_argument(
        "--save_hidden_states",
        action="store_true",
        help="Override config to enable hidden-states extraction (stub — no-op for now)",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.save_hidden_states:
        cfg.analysis.save_hidden_states = True

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/runs") / timestamp
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    cfg.output_dir = run_dir

    # Save a snapshot of the full config for reproducibility
    (run_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir}")

    ensure_hf_token()
    torch_dtype = resolve_torch_dtype(cfg.model.dtype)

    # Stage 1: fetch prompts
    prompts_path = fetch_prompt_sample(
        dataset_name=cfg.dataset.name,
        subset=cfg.dataset.subset,
        split=cfg.dataset.split,
        prompt_field=cfg.dataset.prompt_field,
        max_samples=cfg.dataset.max_samples,
        shuffle=cfg.dataset.shuffle,
        seed=cfg.dataset.seed,
        output_path=run_dir / "data" / "prompts.jsonl",
    )

    # Stage 2: generate and score
    prompt_records = read_jsonl(prompts_path)[: cfg.dataset.max_prompts]
    generations_path = run_prompt_batch(
        prompt_records=prompt_records,
        model_name=cfg.model.name,
        device_map="auto",
        torch_dtype=torch_dtype,
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        chunk_size=cfg.analysis.chunk_size,
        n_values=cfg.analysis.n_values,
        output_path=run_dir / "data" / "generations.jsonl",
        save_hidden_states=cfg.analysis.save_hidden_states,
    )

    # Stage 3: plot
    generation_records = read_jsonl(generations_path)
    plot_metric_distributions(
        generation_records,
        n_values=cfg.analysis.n_values,
        output_dir=run_dir / "plots",
    )

    print(f"\nRun complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
