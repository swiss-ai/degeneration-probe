"""Subtask K infra: score a trained LoRA onset-probe checkpoint (subtask I/J) on
splits the training run itself never touched.

`scripts/train_onset_lora_pilot.py` only ever evaluates on `val` (see its
`evaluation_results.json` / `eval_metrics.jsonl`). To compare the LoRA condition
against the no-LoRA condition (subtask F, which covers `val` / `test_indomain` /
`test_heldout_domains`) on equal footing, this script loads a saved checkpoint
(base model + its LoRA adapter + its probe head, via `setup_probe(load_from="disk")`)
and scores it on `test_indomain` and/or `test_heldout_domains`.

Eval cost control: a full forward pass over every rollout in a split would be
expensive (test_indomain has 3598 rollouts, test_heldout_domains 6300, run through
the full 8B model). Instead this reuses subtask G's `select_lora_pilot_rollouts`
-- every positive rollout plus a domain-stratified negative subsample -- sized so
the positive:negative ratio matches what `train_onset_lora_pilot.py` used for its
own `val` split (~3.5 negatives per positive), keeping per-checkpoint eval cost in
the same ballpark as the pilot run's own val-scoring step rather than a much larger
new commitment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from degeneration_probe.config import ProbeConfig
from degeneration_probe.data.dataset import TokenizedProbingDataset, TokenizedProbingDatasetConfig, tokenized_probing_collate_fn
from degeneration_probe.data.onset_dataset import build_rollout_index
from degeneration_probe.data.onset_lora_dataset import build_onset_probing_items, select_lora_pilot_rollouts
from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.evaluation.evaluate import evaluate_probe
from degeneration_probe.probes.value_head_probe import setup_probe
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"
DEFAULT_SPLITS = ("test_indomain", "test_heldout_domains")
NEGATIVE_PER_POSITIVE = 3.5  # matches the pilot's own val split ratio (~393/113)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=str, default=str(DEFAULT_DATASET_CONFIG_PATH))
    parser.add_argument("--probe-id", type=str, required=True, help="Checkpoint dir name under LOCAL_PROBES_DIR, e.g. onset_multi_horizon_lora_pilot_layer16")
    parser.add_argument("--splits", nargs="*", type=str, default=list(DEFAULT_SPLITS))
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Defaults to <probe_path>/eval_extra_splits.parquet")
    args = parser.parse_args()

    dataset_config = DatasetGenConfig.from_yaml(args.dataset_config)

    # Read the checkpoint's own training config so this eval mirrors training exactly
    # (probe_dtype, normalize_before_head, horizons, ...) rather than re-guessing them.
    probe_config_stub = ProbeConfig(probe_id=args.probe_id, model_name=dataset_config.model_name, load_from="disk")
    training_config_path = probe_config_stub.probe_path / "training_config.json"
    training_config = json.loads(training_config_path.read_text())
    saved_probe_cfg = training_config["probe_config"]
    onset_horizons = training_config["onset_horizons"]
    onset_horizon_weights = training_config["onset_horizon_weights"]
    layer = saved_probe_cfg["layer"]

    print(f"[{args.probe_id}] layer={layer} horizons={onset_horizons}")

    probe_config = ProbeConfig(
        probe_id=args.probe_id,
        model_name=dataset_config.model_name,
        layer=layer,
        load_from="disk",
        probe_dtype=saved_probe_cfg["probe_dtype"],
        normalize_before_head=saved_probe_cfg["normalize_before_head"],
        attention_probe_n_heads=saved_probe_cfg["attention_probe_n_heads"],
        probe_head_type=saved_probe_cfg["probe_head_type"],
        n_outputs=saved_probe_cfg["n_outputs"],
    )

    print(f"Loading base model: {probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        probe_config.model_name, torch_dtype=resolve_torch_dtype(training_config["model_dtype"])
    )
    model, probe = setup_probe(model, probe_config, seed=args.seed)
    probe.eval()

    onset_labels_df = pd.read_parquet(paths.onset_labels_path(dataset_config))

    all_rows = []
    for split in args.splits:
        split_index = build_rollout_index(onset_labels_df, split)
        n_pos = int(split_index["is_positive"].sum())
        n_neg_target = round(NEGATIVE_PER_POSITIVE * max(n_pos, 1))
        selected = select_lora_pilot_rollouts(split_index, n_negative_target=n_neg_target, seed=args.seed)
        print(f"[{split}] {len(selected)} rollouts selected ({n_pos} positive, {len(selected) - n_pos} negative of {len(split_index) - n_pos} available)")

        items = build_onset_probing_items(dataset_config, selected)
        ds_config = TokenizedProbingDatasetConfig(
            dataset_id=f"onset_lora_{split}", hf_repo="local", split=split,
            max_length=args.max_length, max_completion_length=args.max_completion_length,
            shuffle=False, seed=args.seed,
        )
        dataset = TokenizedProbingDataset(items=items, config=ds_config, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=tokenized_probing_collate_fn)

        metrics = evaluate_probe(
            probe=probe, eval_dataloader=dataloader, task="onset_multi_horizon",
            onset_horizons=onset_horizons, onset_horizon_weights=onset_horizon_weights,
            metric_key_prefix=split, verbose=True, save_roc_curves=False,
        )

        for horizon in onset_horizons:
            all_rows.append(
                {
                    "probe_id": args.probe_id, "layer": layer, "split": split, "horizon": horizon,
                    "design": "multi_horizon" if len(onset_horizons) > 1 else "probe_N",
                    "auc": metrics[f"{split}/horizon_{horizon}_auc"],
                    "ap": metrics[f"{split}/horizon_{horizon}_ap"],
                    "n": metrics[f"{split}/horizon_{horizon}_n"],
                    "n_pos": metrics[f"{split}/horizon_{horizon}_n_pos"],
                    "n_rollouts": len(selected), "n_rollouts_pos": n_pos,
                }
            )

    out_df = pd.DataFrame.from_records(all_rows)
    out_path = Path(args.out) if args.out else (probe_config.probe_path / "eval_extra_splits.parquet")
    out_df.to_parquet(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
