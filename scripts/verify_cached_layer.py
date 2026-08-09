"""Confirm the activation cache holds what the live probe would have computed.

Two things can go wrong between a cached vector and a live one, and both are
silent. The probed layer sits one slot later in the cache than its number,
because the stored tensor keeps the embedding output first, so reading one slot
off trains a perfectly convergent probe on the wrong layer. And the cache was
written from a prompt rebuilt by the generation pipeline, so any drift between
that and the prompt training builds would shift every hidden state without
changing a single shape.

This checks both against the real model, and checks that the comparison has
power: the neighbouring slots must disagree, or matching proves nothing.

    python scripts/verify_cached_layer.py --run-dir <dir> --rollouts 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.config import ExperimentConfig
from degeneration_probe.data.activation_store import (
    cached_slot_for_probe_layer,
    load_probe_layer,
)
from degeneration_probe.data.dataset import DegenerationTokenDataset, DegenerationRecord
from degeneration_probe.probes.linear_probe import setup_probe
from degeneration_probe.utils.file_utils import load_json
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=4)
    args = parser.parse_args()

    config = ExperimentConfig.from_dict(load_json(args.run_dir / "resolved_config.json"))
    probe_layer = config.training.probe.layer
    slot = cached_slot_for_probe_layer(probe_layer)
    build_root = config.dataset.build_root
    print(f"probed layer {probe_layer} -> cached slot {slot}")

    onset = pd.read_parquet(Path(build_root) / "onset_labels" / "onset_labels.parquet")
    prompts = pd.read_parquet(Path(build_root) / "prompts" / "prompts.parquet")
    sample = onset.sample(n=args.rollouts, random_state=0)

    model, tokenizer = load_model_and_tokenizer(
        config.model.name,
        tokenizer_name=config.model.tokenizer_name,
        torch_dtype=resolve_torch_dtype(config.model.dtype),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    # No adapters: the cache describes the unadapted model.
    config.training.lora.enabled = False
    _, probe = setup_probe(model, config.training.probe, config.training.lora)

    prompt_text = dict(zip(prompts["prompt_id"], prompts["prompt_text"]))
    failures = 0
    for row in sample.itertuples():
        generations = pd.read_parquet(
            Path(build_root) / "generations" / row.domain / "shard_00000.parquet",
            columns=["prompt_id", "rollout_idx", "generated_token_ids"],
        )
        match = generations[
            (generations["prompt_id"] == row.prompt_id)
            & (generations["rollout_idx"] == row.rollout_idx)
        ]
        token_ids = list(match["generated_token_ids"].iloc[0])

        record = DegenerationRecord(
            prompt_id=row.prompt_id,
            rollout_idx=int(row.rollout_idx),
            domain=row.domain,
            split=row.split,
            prompt_text=prompt_text[row.prompt_id],
            generated_token_ids=token_ids,
            targets=[0.0] * len(token_ids),
        )
        item = DegenerationTokenDataset(
            [record], tokenizer, config.dataset.tokenization
        )[0]

        with torch.no_grad():
            probe(
                input_ids=item["input_ids"].unsqueeze(0).to(probe.device),
                attention_mask=item["attention_mask"].unsqueeze(0).to(probe.device),
            )
        live = probe._captured[0, item["prompt_length"] :].cpu()

        cached = load_probe_layer(
            build_root, row.domain, row.prompt_id, int(row.rollout_idx), probe_layer=probe_layer
        )[: live.shape[0]]
        live = live[: cached.shape[0]]

        agreement = cosine(cached, live)
        neighbours = {}
        for offset in (-1, +1):
            other = cached_slot_for_probe_layer(probe_layer + offset)
            if 0 <= other:
                try:
                    alternative = load_probe_layer(
                        build_root,
                        row.domain,
                        row.prompt_id,
                        int(row.rollout_idx),
                        probe_layer=probe_layer + offset,
                    )[: live.shape[0]]
                    neighbours[probe_layer + offset] = cosine(alternative, live)
                except ValueError:
                    pass

        passed = agreement > 0.999 and all(value < 0.99 for value in neighbours.values())
        failures += not passed
        print(
            f"  {row.domain}/{row.prompt_id}/{row.rollout_idx}: "
            f"tokens={live.shape[0]} cosine(layer {probe_layer})={agreement:.6f} "
            + " ".join(f"cosine(layer {k})={v:.4f}" for k, v in sorted(neighbours.items()))
            + ("  OK" if passed else "  MISMATCH")
        )

    if failures:
        raise SystemExit(
            f"{failures} rollout(s) disagreed: the cache does not hold what the live probe reads"
        )
    print("\nThe cache matches the live capture, and the neighbouring layers do not.")


if __name__ == "__main__":
    main()
