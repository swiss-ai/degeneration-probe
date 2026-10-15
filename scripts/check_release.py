"""Check that the released probes reproduce the scores the paper evaluated.

Every probe folder written by scripts/export_release.py is loaded with the
standalone loader in release/onward_probe.py, which shares no code with
training, and scores a sample of validation responses of its generator. The
scores are compared token by token with the ones stored by the evaluation of the
probe's source run, and the first alarm at each released threshold is compared
response by response.

    python scripts/check_release.py [--release <store>/release] [--per-class 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "release"))

from onward_probe import DegenerationMonitor  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
DATASETS = {"swiss-ai/Apertus-8B-Instruct-2509": "apertus-8b-instruct",
            "meta-llama/Llama-3.1-8B-Instruct": "llama3p1-8b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct-v0p1"}


def sample(stored: pd.DataFrame, per_class: int, seed: int = 0) -> pd.DataFrame:
    parts = [stored[stored["is_positive"] == flag] for flag in (True, False)]
    return pd.concat([p.sample(min(per_class, len(p)), random_state=seed) for p in parts])


def responses(dataset: str, keys: pd.DataFrame) -> pd.DataFrame:
    root = STORE / f"degeneration-dataset-{dataset}"
    prompts = pd.read_parquet(root / "prompts" / "prompts.parquet", columns=["prompt_id", "prompt_text"])
    wanted = set(zip(keys["prompt_id"], keys["rollout_idx"]))
    shards = []
    for path in sorted((root / "generations").glob("*/*.parquet")):
        shard = pd.read_parquet(path, columns=["prompt_id", "rollout_idx", "generated_token_ids"])
        shards.append(shard[[k in wanted for k in zip(shard["prompt_id"], shard["rollout_idx"])]])
    return pd.concat(shards).merge(prompts, on="prompt_id")


def first_alarm(scores: np.ndarray, tau: float):
    hits = np.flatnonzero(scores >= tau)
    return int(hits[0]) if hits.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=STORE / "release")
    parser.add_argument("--per-class", type=int, default=8)
    args = parser.parse_args()
    report = []
    for repo in sorted(args.release.glob("onward-probes-*")):
        model = tokenizer = None
        for folder in sorted(p for p in repo.iterdir() if (p / "config.json").is_file()):
            config = json.loads((folder / "config.json").read_text())
            if model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_model_revision"])
                model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["base_model_revision"],
                                                             torch_dtype=torch.bfloat16).to("cuda")
            monitor = DegenerationMonitor.from_pretrained(str(folder), model=model, tokenizer=tokenizer)
            stored = pd.read_parquet(STORE / "outputs" / config["source_run"] / "selected" / f"layer_{config['layer']:02d}"
                                     / "scores" / "val.parquet")
            picked = sample(stored, args.per_class)
            texts = responses(DATASETS[config["base_model"]], picked)
            diffs, agree, total = [], 0, 0
            for _, row in picked.merge(texts, on=["prompt_id", "rollout_idx"]).iterrows():
                reference = np.asarray(row["scores"], dtype=np.float32)
                generated = list(row["generated_token_ids"])[: len(reference)]
                ours = monitor.score(monitor.build_prompt_ids(row["prompt_text"]), generated).numpy()
                diffs.append(np.abs(ours - reference))
                for entry in monitor.thresholds["thresholds"]:
                    total += 1
                    agree += first_alarm(ours, entry["tau"]) == first_alarm(reference, entry["tau"])
            diffs = np.concatenate(diffs)
            report.append({"probe": f"{repo.name}/{folder.name}", "responses": len(picked), "tokens": diffs.size,
                           "max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean()),
                           "p99_abs_diff": float(np.quantile(diffs, 0.99)), "first_alarm_agreement": agree / total})
            print(report[-1], flush=True)
            model = monitor.close()
        del model
        torch.cuda.empty_cache()
    print(pd.DataFrame(report).to_string(index=False))


if __name__ == "__main__":
    main()
