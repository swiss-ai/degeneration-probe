"""Evaluate a saved degeneration probe checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from degeneration_probe.dataset import DegenerationDataset, make_collate_fn
from degeneration_probe.model_utils import load_model_and_tokenizer, resolve_torch_dtype
from degeneration_probe.probe import SequenceProbe
from degeneration_probe.evaluation import evaluate_and_save

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a degeneration probe checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to probe checkpoint dir")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation JSONL")
    parser.add_argument("--model_name", type=str, default=None, help="Model name (reads from checkpoint config if omitted)")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save results (defaults to checkpoint/eval)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    with open(ckpt_path / "probe_config.json") as f:
        probe_cfg = json.load(f)

    # Determine output dir
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path / "eval"

    # Load model
    model_name = args.model_name
    if model_name is None:
        # Try to read from the training config saved alongside the checkpoint
        train_config_path = ckpt_path.parent / "config.yaml"
        if train_config_path.exists():
            import yaml
            with open(train_config_path) as f:
                train_cfg = yaml.safe_load(f)
            model_name = train_cfg.get("model_name")
    if model_name is None:
        parser.error("Could not determine model_name. Pass --model_name explicitly.")

    log.info("Loading model: %s", model_name)
    torch_dtype = resolve_torch_dtype(args.model_dtype)
    model, tokenizer = load_model_and_tokenizer(model_name, torch_dtype=torch_dtype)

    # Load probe
    log.info("Loading probe from: %s", ckpt_path)
    probe = SequenceProbe.load(ckpt_path, model)

    # Load eval data
    eval_dataset = DegenerationDataset(args.eval_data)
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    log.info("Evaluating on %d samples...", len(eval_dataset))
    metrics = evaluate_and_save(
        probe, eval_loader, output_dir, threshold=probe_cfg.get("threshold", 0.5),
    )

    log.info("Results saved to %s", output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
