"""Unified CLI: python -m degeneration_probe {generate,train,evaluate}."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


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
    if args.save_hidden_states:
        cfg.analysis.save_hidden_states = True

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
        save_hidden_states=cfg.analysis.save_hidden_states,
    )

    generation_records = read_jsonl(generations_path)
    plot_metric_distributions(
        generation_records,
        n_values=cfg.analysis.n_values,
        output_dir=run_dir / "plots",
    )

    print(f"\nRun complete. Results saved to: {run_dir}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, random_split

    from degeneration_probe.config import load_training_config
    from degeneration_probe.data import resolve_model_from_data
    from degeneration_probe.dataset import DegenerationDataset, make_collate_fn
    from degeneration_probe.evaluation import evaluate_and_save
    from degeneration_probe.model_utils import load_model_and_tokenizer
    from degeneration_probe.probe import SequenceProbe
    from degeneration_probe.training import train

    cfg = load_training_config(Path(args.config))

    if not cfg.train_data:
        raise SystemExit("Error: 'train_data' is empty in config. Provide at least one JSONL path.")

    # Auto-resolve model from data
    model_name = resolve_model_from_data(cfg.train_data)
    log.info("Auto-resolved model from data: %s", model_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot (include resolved model_name for evaluate)
    config_snapshot = dataclasses.asdict(cfg)
    config_snapshot["model_name"] = model_name
    (output_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, default=str),
        encoding="utf-8",
    )

    torch.manual_seed(cfg.seed)

    log.info("Loading model: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)

    log.info("Creating probe on layer %d, pooling=%s", cfg.probe.layer, cfg.probe.pooling)
    probe = SequenceProbe(
        model=model,
        layer_idx=cfg.probe.layer,
        pooling=cfg.probe.pooling,
        seed=cfg.seed,
    )

    log.info("Loading training data: %s", cfg.train_data)
    full_dataset = DegenerationDataset(cfg.train_data)
    collate_fn = make_collate_fn(tokenizer, max_length=cfg.max_length)

    if cfg.eval_data is not None:
        train_dataset = full_dataset
        eval_dataset = DegenerationDataset(cfg.eval_data)
    else:
        n_eval = max(1, int(len(full_dataset) * cfg.eval_fraction))
        n_train = len(full_dataset) - n_eval
        train_dataset, eval_dataset = random_split(
            full_dataset,
            [n_train, n_eval],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    log.info("Train: %d samples, Eval: %d samples", len(train_dataset), len(eval_dataset))

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"probe_{ts}",
            config=config_snapshot,
        )

    train(
        probe,
        train_loader,
        eval_loader,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        pos_weight=cfg.pos_weight,
        use_wandb=use_wandb,
    )

    log.info("Running final evaluation...")
    metrics = evaluate_and_save(
        probe, eval_loader, output_dir / "eval", threshold=cfg.probe.threshold,
    )
    log.info("Final metrics: %s", metrics)

    probe.save(output_dir / "checkpoint")
    log.info("Probe saved to %s", output_dir / "checkpoint")

    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from degeneration_probe.dataset import DegenerationDataset, make_collate_fn
    from degeneration_probe.evaluation import evaluate_and_save
    from degeneration_probe.model_utils import load_model_and_tokenizer, resolve_torch_dtype
    from degeneration_probe.probe import SequenceProbe

    ckpt_path = Path(args.checkpoint)
    with open(ckpt_path / "probe_config.json") as f:
        probe_cfg = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path / "eval"

    # Resolve model name: CLI override > config.json > config.yaml (legacy)
    model_name = args.model_name
    if model_name is None:
        for config_file, loader in [
            (ckpt_path.parent / "config.json", json.load),
            (ckpt_path.parent / "config.yaml", None),
        ]:
            if config_file.exists():
                if loader is json.load:
                    with open(config_file) as f:
                        train_cfg = json.load(f)
                else:
                    import yaml
                    with open(config_file) as f:
                        train_cfg = yaml.safe_load(f)
                model_name = train_cfg.get("model_name")
                if model_name:
                    break

    if model_name is None:
        raise SystemExit(
            "Could not determine model_name from checkpoint config. "
            "Pass --model_name explicitly."
        )

    log.info("Loading model: %s", model_name)
    torch_dtype = resolve_torch_dtype(args.model_dtype)
    model, tokenizer = load_model_and_tokenizer(model_name, torch_dtype=torch_dtype)

    log.info("Loading probe from: %s", ckpt_path)
    probe = SequenceProbe.load(ckpt_path, model)

    eval_dataset = DegenerationDataset(args.eval_data)
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    log.info("Evaluating on %d samples...", len(eval_dataset))
    metrics = evaluate_and_save(
        probe, eval_loader, output_dir, threshold=probe_cfg.get("threshold", 0.5),
    )

    log.info("Results saved to %s", output_dir)
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI backend server."""
    import uvicorn
    from degeneration_probe.server.app import create_app

    app = create_app(db_path=args.db_path)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_worker(args: argparse.Namespace) -> None:
    """Start the inference worker."""
    from degeneration_probe.worker.serve import main as worker_main
    import sys
    # Re-inject args so worker's argparse sees them
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
    p_gen.add_argument("--save_hidden_states", action="store_true")
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
    p_eval.add_argument("--model_dtype", default="auto")
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
