"""Frozen and adapted hidden states at one layer, for the PCA of degenerate and healthy text.

For one model, this samples degenerate and long healthy responses from the
in-domain test split, teacher-forces each through the generator twice, once as
is and once with the LoRA adapters of the adapted probe read at the same layer,
and keeps the residual stream at that layer only. The two conditions see the
same token ids, so any difference between them comes from the adapters.

Healthy responses are drawn from those of at least ``--min-healthy-tokens``
tokens, because flags and representations drift with position and degenerate
responses all run to the token limit. The adapted probe is the seed with the
highest validation warning coverage among the selectable seeds, read at its
selected checkpoint.

The forward pass reuses the functions of scripts/extract_lora_linearity_activations.py,
which built the training-time cache, and the first frozen responses are checked
against that cache before anything is written. Each dimension is standardised
over the tokens first, because in some models a few large, nearly constant
dimensions make neighbouring layers look alike under a raw cosine (Mistral's
layers 4 to 6 agree at 0.99). Standardised, the recomputed layer must agree with
its cached copy above 0.98 and its cached neighbours must stay below 0.95.

    python scripts/extract_pca_states.py --model apertus --layer 15 --out-dir <dir>

With ``--paired`` the healthy responses are siblings: for each sampled
degenerate response, up to ``--siblings`` healthy responses to the same prompt
that run at least as long as its onset, so that the two can be compared at the
same positions with the prompt held fixed.

Writes <out-dir>/<model>/{frozen,adapted}/<domain>/<prompt_id>/rollout_<k>.safetensors
(tensor ``hidden_states``, shape [tokens, hidden], float16) and a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig  # noqa: E402
from degeneration_probe.data.activation_store import load_probe_layer  # noqa: E402
from degeneration_probe.dataset_gen.cache_activations import (  # noqa: E402
    build_completion_input_ids,
    extract_completion_hidden_states,
)
from degeneration_probe.dataset_gen.config import DatasetGenConfig  # noqa: E402
from degeneration_probe.dataset_gen.generate import load_tokenizer_for_generation  # noqa: E402
from degeneration_probe.utils.file_utils import load_json  # noqa: E402
from degeneration_probe.utils.model_utils import resolve_torch_dtype  # noqa: E402
from scripts.extract_lora_linearity_activations import (  # noqa: E402
    attach_text_and_tokens,
    load_rollouts,
)

OUTPUTS = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs")
DATASETS = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
RECIPE = dict(selection="frontier_window", window=512, anchor="centered", label="frontier_hard", horizon=256, loss="bce")


def finished_runs(model: str, lora: str):
    for info_path in sorted(OUTPUTS.glob(f"{DATASETS[model]}_*/latest/run_info.json")):
        info = json.loads(info_path.read_text())
        axes = info.get("axes", {})
        if (info.get("status") == "finished" and axes.get("lora") == lora and axes.get("model") == model
                and all(axes.get(k) == v for k, v in RECIPE.items())):
            yield info_path.parent.resolve(), axes


def adapted_checkpoint(model: str, layer: int) -> Path:
    candidates = []
    for run_dir, axes in finished_runs(model, "all"):
        selection = run_dir / "checkpoint_selection.json"
        if axes.get("layer") != layer or not selection.is_file():
            continue
        depth = json.loads(selection.read_text())["depths"][0]
        candidates.append((bool(depth["selectable"]), float(depth["warning_recall_256"]), run_dir, depth))
    if not candidates:
        raise SystemExit(f"no finished adapted {model} run at layer {layer}")
    _, _, run_dir, depth = max(candidates, key=lambda c: (c[0], c[1]))
    checkpoint = run_dir / f"checkpoint-{depth['selected_step']}"
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise SystemExit(f"{checkpoint} holds no LoRA adapter")
    return checkpoint


def sample_keys(build_root: Path, n_degenerate: int, n_healthy: int, min_healthy_tokens: int, seed: int):
    import pandas as pd

    labels = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    labels = labels[(labels["split"] == "test_indomain") & ~labels["onset_resolution"].isin(["judge_failed", "not_found"])]
    degenerate = labels[labels["is_positive"].astype(bool) & (labels["onset_position"] >= 256)]
    healthy = labels[~labels["is_positive"].astype(bool) & (labels["num_tokens"] >= min_healthy_tokens)]
    rng = np.random.default_rng(seed)
    picks = [frame.iloc[rng.choice(len(frame), size=min(n, len(frame)), replace=False)]
             for frame, n in ((degenerate, n_degenerate), (healthy, n_healthy))]
    return [(r.prompt_id, int(r.rollout_idx)) for frame in picks for r in frame.itertuples()]


def standardised_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine of the two activations after standardising every dimension over the tokens, in float64."""
    a, b = a.double(), b.double()
    a = (a - a.mean(0)) / (a.std(0) + 1e-6)
    b = (b - b.mean(0)) / (b.std(0) + 1e-6)
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def sample_paired_keys(build_root: Path, n_degenerate: int, siblings: int, seed: int):
    import pandas as pd

    labels = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    labels = labels[(labels["split"] == "test_indomain") & ~labels["onset_resolution"].isin(["judge_failed", "not_found"])]
    healthy = labels[~labels["is_positive"].astype(bool)]
    rng = np.random.default_rng(seed)
    degenerate = labels[labels["is_positive"].astype(bool) & (labels["onset_position"] >= 256)]
    degenerate = degenerate.iloc[rng.permutation(len(degenerate))]
    keys, per_prompt = [], {}
    for row in degenerate.itertuples():
        long_enough = healthy[(healthy["prompt_id"] == row.prompt_id) & (healthy["num_tokens"] >= row.onset_position)]
        if len(long_enough) < 2 or per_prompt.get(row.prompt_id, 0) >= 2:
            continue
        per_prompt[row.prompt_id] = per_prompt.get(row.prompt_id, 0) + 1
        keys.append((row.prompt_id, int(row.rollout_idx)))
        chosen = long_enough.iloc[rng.choice(len(long_enough), min(siblings, len(long_enough)), replace=False)]
        keys += [(r.prompt_id, int(r.rollout_idx)) for r in chosen.itertuples()]
        if sum(per_prompt.values()) >= n_degenerate:
            break
    return list(dict.fromkeys(keys))


def write_state(path: Path, hidden: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": hidden.contiguous()}, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(DATASETS), required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS / "paper_figures" / "pca_states")
    parser.add_argument("--n-degenerate", type=int, default=30)
    parser.add_argument("--n-healthy", type=int, default=30)
    parser.add_argument("--min-healthy-tokens", type=int, default=1000)
    parser.add_argument("--sanity-rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paired", action="store_true", help="healthy siblings of the same prompt")
    parser.add_argument("--siblings", type=int, default=3)
    args = parser.parse_args()

    frozen_run = next((run for run, axes in finished_runs(args.model, "none")), None)
    if frozen_run is None:
        raise SystemExit(f"no finished frozen {args.model} run")
    config = ExperimentConfig.from_dict(load_json(frozen_run / "resolved_config.json"))
    build_root = Path(config.dataset.build_root)
    build_config = DatasetGenConfig.from_yaml(config.dataset.build_config)
    checkpoint = adapted_checkpoint(args.model, args.layer)
    print(f"model {args.model}, layer {args.layer}\n  frozen settings from {frozen_run}\n  adapters from {checkpoint}")

    if args.paired:
        keys = sample_paired_keys(build_root, args.n_degenerate, args.siblings, args.seed)
    else:
        keys = sample_keys(build_root, args.n_degenerate, args.n_healthy, args.min_healthy_tokens, args.seed)
    rollouts = attach_text_and_tokens(build_root, load_rollouts(build_root, [], keys))
    print(f"{len(rollouts)} responses, {int(rollouts['is_positive'].sum())} degenerate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_torch_dtype(config.model.dtype, default=torch.bfloat16)
    tokenizer = load_tokenizer_for_generation(build_config)
    model = AutoModelForCausalLM.from_pretrained(build_config.model_name, dtype=dtype).to(device)
    model.eval()
    model.config.use_cache = False
    slot = args.layer + 1  # hidden_states[0] is the embedding output
    out = args.out_dir / args.model

    for condition in ("frozen", "adapted"):
        if condition == "adapted":
            model = PeftModel.from_pretrained(model, checkpoint)
            model.eval()
            model.get_base_model().config.use_cache = False
        for index, row in enumerate(rollouts.itertuples()):
            input_ids, prompt_len = build_completion_input_ids(tokenizer, row.prompt_text, list(row.generated_token_ids))
            hidden = extract_completion_hidden_states(model, input_ids, prompt_len, len(row.generated_token_ids), device)[slot]
            if condition == "frozen" and index < args.sanity_rollouts:
                cached = load_probe_layer(config.dataset.activations_dir, row.domain, row.prompt_id, int(row.rollout_idx),
                                          probe_layer=args.layer)
                n = min(len(cached), len(hidden))
                agreement = standardised_cosine(cached[:n], hidden[:n])
                neighbours = [standardised_cosine(load_probe_layer(config.dataset.activations_dir, row.domain, row.prompt_id,
                                                                int(row.rollout_idx), probe_layer=layer)[:n], hidden[:n])
                              for layer in (args.layer - 1, args.layer + 1) if layer >= 1]
                print(f"  sanity {row.prompt_id} r{row.rollout_idx}: cosine {agreement:.5f}, neighbours "
                      + ", ".join(f"{v:.4f}" for v in neighbours))
                if agreement <= 0.98 or any(v >= 0.95 for v in neighbours):
                    raise SystemExit("recomputed frozen states do not match the activation cache; nothing written")
            write_state(out / condition / row.domain / row.prompt_id / f"rollout_{row.rollout_idx}.safetensors", hidden)
        print(f"  {condition}: {len(rollouts)} responses written")

    manifest = rollouts[["prompt_id", "rollout_idx", "domain", "split", "is_positive", "onset_position", "num_tokens"]].copy()
    manifest["layer"], manifest["adapter_checkpoint"] = args.layer, str(checkpoint)
    manifest.to_parquet(out / "manifest.parquet", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
