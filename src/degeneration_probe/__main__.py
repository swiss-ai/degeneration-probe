"""Unified CLI: python -m degeneration_probe {generate,train,evaluate,serve,worker,ui}."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


from degeneration_probe._fork_path import ensure_fork_on_syspath


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    from degeneration_probe.config import load_config
    from degeneration_probe.data import ensure_hf_token, fetch_prompt_sample, read_jsonl
    from degeneration_probe.generation import run_prompt_batch
    from degeneration_probe.model_utils import resolve_torch_dtype
    from degeneration_probe.plotting import plot_metric_distributions

    cfg = load_config(Path(args.config))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/generations") / timestamp
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    cfg.output_dir = run_dir

    (run_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir}")

    ensure_hf_token()
    torch_dtype = resolve_torch_dtype(cfg.model.dtype)

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
    )

    generation_records = read_jsonl(generations_path)
    plot_metric_distributions(
        generation_records,
        n_values=cfg.analysis.n_values,
        output_dir=run_dir / "plots",
    )

    print(f"\nRun complete. Results saved to: {run_dir}")


# ---------------------------------------------------------------------------
# train  →  delegates to fork/degeneration/train.py
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from degeneration_probe.config import load_training_config
    from degeneration_probe.data import resolve_model_from_data

    ensure_fork_on_syspath()
    from degeneration.train import TrainConfig, train as fork_train

    cfg = load_training_config(Path(args.config))
    if not cfg.train_data and cfg.hf_dataset is None:
        raise SystemExit(
            "Error: provide either 'train_data' (JSONL paths) or 'hf_dataset' in the config."
        )

    if cfg.model is not None:
        model_name = cfg.model.name
        log.info("Using configured model override: %s", model_name)
    elif cfg.train_data:
        model_name = resolve_model_from_data(cfg.train_data)
        log.info("Auto-resolved model from data: %s", model_name)
    else:
        raise SystemExit(
            "Error: when using 'hf_dataset', the model must be set explicitly via 'model.name'."
        )

    train_cfg = TrainConfig(
        model_name=model_name,
        layer=cfg.probe.layer,
        lora_enabled=cfg.probe.lora.enabled,
        lora_rank=cfg.probe.lora.rank,
        lora_alpha=cfg.probe.lora.alpha,
        lora_dropout=cfg.probe.lora.dropout,
        lora_layers=list(cfg.probe.lora.layers) if cfg.probe.lora.layers else None,
        train_data=list(cfg.train_data),
        eval_data=cfg.eval_data,
        hf_dataset=cfg.hf_dataset.name if cfg.hf_dataset else None,
        hf_train_split=cfg.hf_dataset.train_split if cfg.hf_dataset else "train",
        hf_eval_split=cfg.hf_dataset.eval_split if cfg.hf_dataset else "validation",
        hf_max_train_rows=cfg.hf_dataset.max_train_rows if cfg.hf_dataset else None,
        hf_max_eval_rows=cfg.hf_dataset.max_eval_rows if cfg.hf_dataset else None,
        eval_fraction=cfg.eval_fraction,
        max_length=cfg.max_length,
        window_size=cfg.label.window_size,
        primary_n=cfg.label.primary_n,
        head_lr=cfg.learning_rate.head,
        lora_lr=cfg.learning_rate.lora,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
    )
    ckpt = fork_train(train_cfg)
    print(f"Probe saved to {ckpt}")


# ---------------------------------------------------------------------------
# evaluate  →  delegates to fork/degeneration/evaluate.py
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    ensure_fork_on_syspath()
    from degeneration.evaluate import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint,
        eval_data=args.eval_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# serve / worker / ui
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI backend server."""
    import uvicorn
    from degeneration_probe.server.app import create_app

    app = create_app(db_path=args.db_path)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_worker(args: argparse.Namespace) -> None:
    """Start the inference worker."""
    # The worker imports from the fork at load time; make it importable.
    ensure_fork_on_syspath()
    from degeneration_probe.worker.serve import main as worker_main
    sys.argv = ["worker", "--model", args.model, "--host", args.host, "--port", str(args.port)]
    if args.probe:
        sys.argv.extend(["--probe", args.probe])
    if args.dtype:
        sys.argv.extend(["--dtype", args.dtype])
    worker_main()


def cmd_ui(args: argparse.Namespace) -> None:
    """Start the Gradio UI."""
    from degeneration_probe.ui.app import build_ui
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="degeneration_probe",
        description="Degeneration probe research pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate completions and compute metrics")
    p_gen.add_argument("--config", required=True, help="Path to generation YAML config")
    p_gen.set_defaults(func=cmd_generate)

    # train
    p_train = subparsers.add_parser("train", help="Train a degeneration probe")
    p_train.add_argument("--config", required=True, help="Path to training YAML config")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a saved probe checkpoint")
    p_eval.add_argument("--checkpoint", required=True, help="Path to probe checkpoint dir")
    p_eval.add_argument("--eval_data", required=True, help="Path to evaluation JSONL")
    p_eval.add_argument("--model_name", default=None, help="Model name override")
    p_eval.add_argument("--batch_size", type=int, default=4)
    p_eval.add_argument("--max_length", type=int, default=2048)
    p_eval.add_argument("--output_dir", default=None, help="Where to save results")
    p_eval.set_defaults(func=cmd_evaluate)

    sp_serve = subparsers.add_parser("serve", help="Start the FastAPI backend server")
    sp_serve.add_argument("--host", default="0.0.0.0")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--db-path", default="data/degeneration_probe.db")
    sp_serve.set_defaults(func=cmd_serve)

    sp_worker = subparsers.add_parser("worker", help="Start the inference worker")
    sp_worker.add_argument("--model", required=True, help="HuggingFace model name")
    sp_worker.add_argument("--probe", default=None, help="Path to saved probe checkpoint")
    sp_worker.add_argument("--dtype", default=None, help="Model dtype")
    sp_worker.add_argument("--host", default="0.0.0.0")
    sp_worker.add_argument("--port", type=int, default=9000)
    sp_worker.set_defaults(func=cmd_worker)

    sp_ui = subparsers.add_parser("ui", help="Start the Gradio UI")
    sp_ui.add_argument("--host", default="0.0.0.0")
    sp_ui.add_argument("--port", type=int, default=7860)
    sp_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
