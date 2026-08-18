"""Extract teacher-forced, all-layer activations for the frozen-vs-LoRA
representation-linearity comparison notebook.

Two conditions share exactly the same 30 rollouts (three prompts, ten
rollouts each) and exactly the same generated token ids -- the only thing
that differs between them is whether the LoRA adapter is applied:

  - frozen: the base model, adapters disabled.
  - lora:   the same weights with a LoRA adapter loaded on top.

Every rollout is teacher-forced over its already-generated token ids (no
re-generation: the token stream must be identical between conditions so any
difference in the activations is attributable to the adapter, not to
different text). Every decoder layer's residual stream is captured in one
forward pass and written to its own fp16 file, so the pass is paid for once
per condition rather than once per layer.

Prompt reconstruction and the forward pass itself reuse
``degeneration_probe.dataset_gen.cache_activations`` and
``degeneration_probe.dataset_gen.generate`` directly -- the exact functions
that built the training-time cache -- rather than an independently
reimplemented path such as ``DegenerationTokenDataset``. Two things about
that module matter here and are easy to get quietly wrong by reimplementing:
the tokenizer is loaded via ``load_tokenizer_for_generation``, without
``trust_remote_code=True``, and the model is loaded via a plain
``AutoModelForCausalLM.from_pretrained(...)`` with no ``trust_remote_code``
either. Apertus ships natively in this ``transformers`` install; passing
``trust_remote_code=True`` (as this script's own first version did, via
``degeneration_probe.utils.model_utils.load_model_and_tokenizer``) makes
``from_pretrained`` prefer the Hub repo's own remote modeling code over the
native implementation instead, which produced activations that failed the
sanity check below even though the token ids matched exactly.

Files are written in the same layout the training-time activation cache
uses (``<root>/activations/<domain>/<prompt_id>/rollout_<idx>.safetensors``,
tensor key ``hidden_states``, metadata ``layer_order``), so a condition's
directory can be read back with the ordinary
``degeneration_probe.data.activation_store`` helpers -- including their
slot-offset and layer-order checks -- rather than a bespoke loader.

As a sanity check, the frozen condition's layer-15 activations are compared
against the *existing* training-time activation cache for a few rollouts.
Do NOT mix cached activations for one condition with recomputed ones for the
other: if this check fails, the script stops before writing anything else,
rather than producing activations nobody should plot.

Agreement is judged the same way ``scripts/verify_cached_layer.py`` judges
it: flattened cosine similarity above 0.999, with a power check that the
neighbouring cached layers score below 0.99 (otherwise the check would pass
trivially). A strict elementwise ``torch.allclose`` at bf16-scale tolerance
(what ``degeneration_probe.data.activation_store.agrees_with_live_capture``
does) turned out to fail even for a verified-correct reproduction at this
depth: bf16 (the live forward pass) carries only ~7 mantissa bits, and that
rounding compounds over 15 decoder layers into elementwise disagreement
across most of the tensor even while every token's direction stays within a
fraction of a degree of the fp16 cache (cosine 0.9996 average, 0.9656 worst,
measured directly against this exact rollout before adopting this
criterion). Cosine similarity is also the criterion that actually matters
for a linear probe: it reads direction through a LayerNorm, not raw
magnitude, so this is not a loosened check, it is the correctly-scaled one.

    python scripts/extract_lora_linearity_activations.py \
        --frozen-run-dir outputs/apertus-8b-instruct_L1-31_all_tokens_hard1024_bce_lora-none_s42_7790c264/20260814T050453 \
        --lora-checkpoint outputs/apertus-8b-instruct_L15_all_tokens_hard1024_bce_lora-all_s42_b9e9df34/20260816T085625/checkpoint-1600 \
        --out-dir outputs/lora_linearity_probe/activations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.data.activation_store import (
    LAYER_ORDER_LABEL,
    TENSOR_KEY,
    activation_path,
    load_probe_layer,
)
from degeneration_probe.dataset_gen.cache_activations import (
    build_completion_input_ids,
    extract_completion_hidden_states,
)
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.generate import load_tokenizer_for_generation
from degeneration_probe.utils.file_utils import load_json
from degeneration_probe.utils.model_utils import resolve_torch_dtype

# The three validation-split prompts this notebook compares: two in-domain
# prompts with a mix of healthy and degenerate rollouts (so the same figure
# can separate "about the trajectory" from "about the prompt"), one prompt
# with no degenerate rollouts at all. All ten rollouts of each are used.
DEFAULT_PROMPT_IDS = [
    "if_sft_data_verified_00483",  # 5/10 degenerate
    "llama_nemotron_00398",  # 4/10 degenerate
    "deepmath_103k_00001",  # 0/10 degenerate
]
PROBE_LAYER_FOR_SANITY_CHECK = 15


def load_rollouts(build_root: Path, prompt_ids: List[str]) -> pd.DataFrame:
    onset = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    selected = onset[onset["prompt_id"].isin(prompt_ids)].copy()
    missing = set(prompt_ids) - set(selected["prompt_id"])
    if missing:
        raise ValueError(f"prompt id(s) not found in onset labels: {sorted(missing)}")
    if not (selected["split"] == "val").all():
        bad = selected.loc[selected["split"] != "val", ["prompt_id", "split"]]
        raise ValueError(f"every rollout must be in the validation split, found:\n{bad}")
    # is_positive already implies onset_resolution == "ok" in this corpus (checked once,
    # not re-derived here), so a positive rollout always has a usable frontier.
    bad_positive = selected[selected["is_positive"] & (selected["onset_resolution"] != "ok")]
    if not bad_positive.empty:
        raise ValueError(f"positive rollout(s) without a resolved frontier:\n{bad_positive}")
    selected = selected.sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)
    for prompt_id, group in selected.groupby("prompt_id"):
        if len(group) != 10 or sorted(group["rollout_idx"]) != list(range(10)):
            raise ValueError(f"expected rollouts 0..9 for {prompt_id!r}, found {sorted(group['rollout_idx'])}")
    return selected


def attach_text_and_tokens(build_root: Path, rollouts: pd.DataFrame) -> pd.DataFrame:
    prompts = pd.read_parquet(
        build_root / "prompts" / "prompts.parquet", columns=["prompt_id", "prompt_text"]
    )
    frames = []
    for domain, group in rollouts.groupby("domain"):
        generations = pd.read_parquet(
            build_root / "generations" / domain / "shard_00000.parquet",
            columns=["prompt_id", "rollout_idx", "generated_token_ids"],
        )
        merged = group.merge(generations, on=["prompt_id", "rollout_idx"], how="left", validate="one_to_one")
        frames.append(merged)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.merge(prompts, on="prompt_id", how="left", validate="many_to_one")
    if merged["generated_token_ids"].isna().any() or merged["prompt_text"].isna().any():
        raise ValueError("some rollouts could not be joined to their generation or prompt text")
    return merged.sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)


def write_condition_file(
    condition_root: Path,
    *,
    domain: str,
    prompt_id: str,
    rollout_idx: int,
    hidden_states: torch.Tensor,
    num_layers: int,
    hidden_size: int,
) -> Path:
    path = activation_path(condition_root, domain, prompt_id, rollout_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {TENSOR_KEY: hidden_states.contiguous()},
        str(path),
        metadata={
            "layer_order": LAYER_ORDER_LABEL,
            "prompt_id": prompt_id,
            "domain": domain,
            "rollout_idx": str(rollout_idx),
            "num_layers": str(num_layers),
            "hidden_size": str(hidden_size),
        },
    )
    return path


def cosine_agreement(cached: torch.Tensor, captured: torch.Tensor) -> float:
    """Flattened cosine similarity, the same measure verify_cached_layer.py uses."""
    return float(
        torch.nn.functional.cosine_similarity(
            cached.float().flatten(), captured.float().flatten(), dim=0
        )
    )


def run_condition(
    *,
    condition: str,
    model,
    tokenizer,
    rollouts: pd.DataFrame,
    device: torch.device,
    out_dir: Path,
    activations_root: Path,
    sanity_rollouts: int,
) -> Path:
    condition_root = out_dir / condition
    base_config = model.get_base_model().config if isinstance(model, PeftModel) else model.config
    num_layers = base_config.num_hidden_layers
    hidden_size = base_config.hidden_size
    checked = 0
    for row in rollouts.itertuples():
        input_ids, prompt_len = build_completion_input_ids(
            tokenizer, row.prompt_text, list(row.generated_token_ids)
        )
        completion_len = len(row.generated_token_ids)
        # [num_layers + 1, completion_len, hidden_size], float16, CPU -- identical
        # contract to cache_activations.extract_completion_hidden_states, which is
        # exactly what this is.
        hidden_states = extract_completion_hidden_states(model, input_ids, prompt_len, completion_len, device)
        expected_slots = num_layers + 1
        if hidden_states.shape[0] != expected_slots:
            raise RuntimeError(
                f"model produced {hidden_states.shape[0]} hidden-state slots, "
                f"expected {expected_slots} ({num_layers} layers + embeddings)"
            )

        if condition == "frozen" and checked < sanity_rollouts:
            slot = PROBE_LAYER_FOR_SANITY_CHECK + 1  # EMBEDDING_SLOTS offset, matches activation_store
            captured = hidden_states[slot]
            n_tokens = captured.shape[0]
            cached = load_probe_layer(
                activations_root,
                row.domain,
                row.prompt_id,
                int(row.rollout_idx),
                probe_layer=PROBE_LAYER_FOR_SANITY_CHECK,
            )[:n_tokens]
            captured_n = captured[: cached.shape[0]]
            agreement = cosine_agreement(cached, captured_n)

            # Power check: the neighbouring cached layers must NOT also agree this well,
            # or the comparison proves nothing (e.g. a slot-offset bug that happens to
            # still look plausible).
            neighbour_scores = {}
            for offset in (-1, 1):
                neighbour_layer = PROBE_LAYER_FOR_SANITY_CHECK + offset
                if neighbour_layer < 0:
                    continue
                neighbour_cached = load_probe_layer(
                    activations_root, row.domain, row.prompt_id, int(row.rollout_idx),
                    probe_layer=neighbour_layer,
                )[:n_tokens]
                neighbour_scores[neighbour_layer] = cosine_agreement(neighbour_cached, captured_n)

            passed = agreement > 0.999 and all(score < 0.99 for score in neighbour_scores.values())
            neighbour_str = " ".join(f"cosine(layer {k})={v:.4f}" for k, v in sorted(neighbour_scores.items()))
            print(
                f"  sanity check {'OK' if passed else 'FAILED'}: "
                f"{row.domain}/{row.prompt_id}/rollout_{row.rollout_idx} "
                f"cosine(layer {PROBE_LAYER_FOR_SANITY_CHECK})={agreement:.6f} {neighbour_str}"
            )
            if not passed:
                raise SystemExit(
                    f"Sanity check failed: recomputed frozen activations for "
                    f"{row.domain}/{row.prompt_id}/rollout_{row.rollout_idx} at layer "
                    f"{PROBE_LAYER_FOR_SANITY_CHECK} do not agree with the existing training-time "
                    "cache (cosine > 0.999, with neighbouring layers below 0.99). Stopping before "
                    "writing or plotting anything."
                )
            checked += 1

        path = write_condition_file(
            condition_root,
            domain=row.domain,
            prompt_id=row.prompt_id,
            rollout_idx=int(row.rollout_idx),
            hidden_states=hidden_states,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        print(
            f"  [{condition}] {row.domain}/{row.prompt_id}/rollout_{row.rollout_idx}: "
            f"{hidden_states.shape[1]} tokens -> {path}"
        )
    if condition == "frozen" and checked < sanity_rollouts:
        raise SystemExit(
            f"Only {checked} of {sanity_rollouts} requested sanity-check rollouts were frozen "
            "positives with cached activations available -- widen --sanity-rollouts or check "
            "the rollout selection."
        )
    return condition_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-run-dir", type=Path, required=True, help="Attempt dir of the frozen (lora-none) run, used for build_root/model settings.")
    parser.add_argument("--lora-checkpoint", type=Path, required=True, help="Checkpoint dir of the LoRA run holding adapter_model.safetensors.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "lora_linearity_probe" / "activations")
    parser.add_argument("--prompt-ids", nargs="+", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--sanity-rollouts", type=int, default=2)
    args = parser.parse_args()

    config = ExperimentConfig.from_dict(load_json(args.frozen_run_dir / "resolved_config.json"))
    build_root = Path(config.dataset.build_root)
    # Where the cached hidden states actually live -- usually build_root/activations,
    # but a build can point this elsewhere (e.g. a separate filesystem), so it is
    # read through the dataset config rather than assumed from build_root.
    activations_root = config.dataset.activations_dir
    build_config = DatasetGenConfig.from_yaml(config.dataset.build_config)

    print(f"Loading rollouts for prompts {args.prompt_ids} from {build_root}")
    rollouts = load_rollouts(build_root, args.prompt_ids)
    rollouts = attach_text_and_tokens(build_root, rollouts)
    print(f"{len(rollouts)} rollouts total, {int(rollouts['is_positive'].sum())} positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_torch_dtype(config.model.dtype, default=torch.bfloat16)
    print(f"Loading tokenizer + base model {build_config.model_name} (dtype={dtype}, device={device}) ...")
    tokenizer = load_tokenizer_for_generation(build_config)
    model = AutoModelForCausalLM.from_pretrained(build_config.model_name, dtype=dtype).to(device)
    model.eval()
    model.config.use_cache = False

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Condition A: frozen (adapters disabled) ===")
    run_condition(
        condition="frozen",
        model=model,
        tokenizer=tokenizer,
        rollouts=rollouts,
        device=device,
        out_dir=args.out_dir,
        activations_root=activations_root,
        sanity_rollouts=args.sanity_rollouts,
    )

    print(f"\nLoading LoRA adapter from {args.lora_checkpoint} ...")
    model = PeftModel.from_pretrained(model, args.lora_checkpoint)
    model.eval()
    model.get_base_model().config.use_cache = False

    print("\n=== Condition B: lora (adapter loaded) ===")
    run_condition(
        condition="lora",
        model=model,
        tokenizer=tokenizer,
        rollouts=rollouts,
        device=device,
        out_dir=args.out_dir,
        activations_root=activations_root,
        sanity_rollouts=0,
    )

    manifest = rollouts[
        ["prompt_id", "rollout_idx", "domain", "split", "is_positive", "onset_position", "stop_reason", "num_tokens"]
    ].copy()
    manifest["frozen_run_dir"] = str(args.frozen_run_dir)
    manifest["lora_checkpoint"] = str(args.lora_checkpoint)
    manifest_path = args.out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    print(f"\nWrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
