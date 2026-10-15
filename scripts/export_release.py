"""Write the released probes as one folder per generator, ready to upload to the Hub.

Each generator gets three probes, the ones the paper reports: the frozen probe
at its selected layer, the adapted probe at the same layer and the adapted probe
at its own selected layer. Of the three seeds of each, the one with the highest
validation warning coverage at the 1% budget (the selection rule's own figure)
is released, and the metrics of all three seeds are written next to it.

    <out>/onward-probes-<generator>/<frozen|adapted>-L<layer>/
        probe.safetensors   layer norm and linear head (norm.*, linear.*)
        config.json         base model and revision, layer, hidden size, source run
        thresholds.json     thresholds at 1%, 5% and 10%, persistence 1, calibrated on validation
        seeds.json          per-seed metrics on validation and both test splits
        adapter_model.safetensors, adapter_config.json   adapted probes only

    python scripts/export_release.py [--out <store>/release]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
OUTPUTS = STORE / "outputs"
RECIPE = "frontier_window W=512 centered K=256"
GENERATORS = {
    "apertus": ("apertus-8b-instruct", "swiss-ai/Apertus-8B-Instruct-2509", "b946d40447b2b597999b9c86d44bee0b452c919f",
                [("none", 15), ("all", 15), ("all", 30)]),
    "llama3p1": ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659",
                 [("none", 15), ("all", 15), ("all", 6)]),
    "mistral": ("mistral-7b-instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca",
                [("none", 5), ("all", 5), ("all", 11)]),
}
METRICS = ["negative_fpr", "rollout_auc", "in_pattern_recall", "warning_recall_256", "preonset_coverage_256",
           "fired_before_frontier", "median_offset", "detected_positives", "positives"]


def head_weights(folder: Path) -> tuple[dict, dict]:
    """The layer norm and linear head of a checkpoint, under the names the release loader expects."""
    config = json.loads((folder / "probe_config.json").read_text())
    assert config["architecture"] == "scalar_linear" and config["context_window_size"] == 1, config
    assert config["normalization"] == "layernorm", config
    norm = torch.load(folder / "normalization.bin", map_location="cpu", weights_only=True)
    linear = torch.load(folder / "probe.bin", map_location="cpu", weights_only=True)
    tensors = {"norm.weight": norm["weight"], "norm.bias": norm["bias"],
               "linear.weight": linear["weight"], "linear.bias": linear["bias"]}
    return {name: tensor.float().contiguous() for name, tensor in tensors.items()}, config


def seed_metrics(rows: pd.DataFrame) -> list:
    out = []
    for (seed, run, attempt, step), part in rows.groupby(["seed", "run_name", "attempt", "selected_step"]):
        entry = {"seed": int(seed), "run": run, "attempt": attempt, "checkpoint_step": int(step), "metrics": {}}
        for (split, budget), cell in part.groupby(["split", "target_negative_fpr"]):
            values = cell.iloc[0]
            entry["metrics"].setdefault(split, {})[f"{budget:g}"] = {
                metric: (None if pd.isna(values[metric]) else float(values[metric])) for metric in METRICS}
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=STORE / "release")
    args = parser.parse_args()
    table = pd.read_csv(OUTPUTS / "analysis" / "selected_metrics.csv")
    table = table[table["recipe"].str.startswith(RECIPE) & (table["persistence"] == 1)]
    for model, (slug, base, revision, probes) in GENERATORS.items():
        repo = args.out / f"onward-probes-{slug}"
        for lora, layer in probes:
            rows = table[(table["model"] == model) & (table["lora"] == lora) & (table["layer"] == layer)]
            val = rows[(rows["split"] == "val") & np.isclose(rows["target_negative_fpr"], 0.01)]
            best = val.sort_values("warning_recall_256", ascending=False).iloc[0]
            run = OUTPUTS / best["run_name"] / best["attempt"]
            checkpoint = run / f"checkpoint-{int(best['selected_step'])}"
            source = checkpoint / f"layer_{layer:02d}" if lora == "none" else checkpoint
            regime = "frozen" if lora == "none" else "adapted"
            folder = repo / f"{regime}-L{layer}"
            folder.mkdir(parents=True, exist_ok=True)

            weights, probe_config = head_weights(source)
            assert probe_config["layer_idx"] == layer, (probe_config, layer)
            save_file(weights, str(folder / "probe.safetensors"))
            if lora != "none":
                for name in ["adapter_model.safetensors", "adapter_config.json"]:
                    shutil.copyfile(checkpoint / name, folder / name)
            thresholds = json.loads((run / "selected" / f"layer_{layer:02d}" / "decision_thresholds.json").read_text())
            assert thresholds["persistence"] == 1 and thresholds["tuned_on"] == "val", thresholds
            (folder / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
            config = {
                "base_model": base, "base_model_revision": revision,
                "layer": layer, "hidden_states_index": layer + 1, "hook_module": f"model.layers.{layer}",
                "hidden_size": int(probe_config["input_size"]), "normalization": "layernorm",
                "adapted": lora != "none", "regime": regime,
                "training": {"selection": "frontier_window", "window": 512, "anchor": "centered",
                             "target": "hard", "horizon": 256, "loss": "bce", "steps": 3000, "tokens_per_step": 4096},
                "prompt_format": "chat template without the beginning-of-sequence token, tokenized without special tokens",
                "seed": int(best["seed"]), "checkpoint_step": int(best["selected_step"]),
                "source_run": f"{best['run_name']}/{best['attempt']}",
            }
            (folder / "config.json").write_text(json.dumps(config, indent=2))
            (folder / "seeds.json").write_text(json.dumps(seed_metrics(rows), indent=2))
            print(f"{folder.relative_to(args.out)}: seed {config['seed']}, step {config['checkpoint_step']}")


if __name__ == "__main__":
    main()
