"""Add the rollouts a scored run is missing to its score tables.

Evaluation now keeps capped rollouts the judge ruled not degenerating, as
healthy answers, where it used to drop them. Rescoring a whole split to add a
handful of rollouts would repeat hours of work, so this scores only the
rollouts of a split that the run's existing score tables do not contain, with
the same checkpoint at every depth, and appends them. The thresholds were
frozen on validation, which holds no such rollout, so only the test splits need
re-evaluating afterwards.

    python scripts/score_extra_rollouts.py --run-dir outputs/<run>/<attempt> [--split test_indomain]

Works for both regimes: a frozen run scores all its selected depths in one pass
over the activation cache, an adapted run loads its model and adapters once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import write_scores  # noqa: E402
from degeneration_probe.probes.linear_probe import setup_cached_probe, setup_probe  # noqa: E402
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype  # noqa: E402
from scripts.score_rollouts import load_run_config, rollout_metadata, score_split, score_split_by_layer  # noqa: E402
from scripts.score_selected_depths import assemble_composite, read_selection  # noqa: E402

KEYS = ["prompt_id", "rollout_idx"]


def missing_keys(config, depth_dirs, split: str) -> pd.DataFrame:
    """Rollouts the evaluation population of ``split`` holds that a score table lacks."""
    onset = pd.read_parquet(Path(config.dataset.build_root) / "onset_labels" / "onset_labels.parquet")
    onset = onset[onset["split"] == split]
    wanted = onset[(onset["stop_reason"] != "eos") & onset["onset_resolution"].eq("not_degenerating")][KEYS]
    present = None
    for depth in depth_dirs:
        keys = pd.read_parquet(depth / "scores" / f"{split}.parquet", columns=KEYS)
        present = keys if present is None else present.merge(keys, on=KEYS)
    if present is None:
        return wanted.iloc[:0]
    merged = wanted.merge(present, on=KEYS, how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"][KEYS].reset_index(drop=True)


def append(depth: Path, split: str, frame: pd.DataFrame) -> int:
    path = depth / "scores" / f"{split}.parquet"
    old = pd.read_parquet(path)
    combined = pd.concat([old, frame[old.columns]], ignore_index=True).drop_duplicates(KEYS, keep="first")
    write_scores(combined, path)
    return len(combined) - len(old)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test_indomain")
    parser.add_argument("--subdir", default="selected")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_run_config(run_dir)
    depth_dirs = sorted(p for p in (run_dir / args.subdir).glob("layer_*") if (p / "scores" / f"{args.split}.parquet").is_file())
    if not depth_dirs:
        raise SystemExit(f"no scored depths under {run_dir / args.subdir}")
    keys = missing_keys(config, depth_dirs, args.split)
    if keys.empty:
        print(f"{run_dir.name}: nothing to add to {args.split}")
        return
    keys_path = run_dir / args.subdir / f"extra_{args.split}_keys.parquet"
    keys.to_parquet(keys_path, index=False)
    config.dataset.sampling.restrict_to = str(keys_path)
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    metadata = rollout_metadata(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.training.features.regime == "cached":
        steps = read_selection(run_dir / "checkpoint_selection.json")
        layers = sorted(int(p.name.split("_")[1]) for p in depth_dirs)
        config.training.probe.layers = layers
        composite = assemble_composite(run_dir, {layer: steps[layer] for layer in layers}, "checkpoint-selected")
        hidden = int(list(pd.read_parquet(config.dataset.activations_manifest_path, columns=["shape"])["shape"].iloc[0])[-1])
        probe = setup_cached_probe(config.training.probe, hidden_size=hidden, checkpoint_path=composite,
                                   seed=config.training.runtime.seed, device=device)
        frames = score_split_by_layer(probe, config, None, split=args.split, metadata=metadata, batch_size=1, layers=layers)
    else:
        if len(depth_dirs) != 1:
            raise SystemExit("an adapted run is expected to hold one scored depth")
        provenance = json.loads((depth_dirs[0] / "scoring_provenance.json").read_text())
        model, tokenizer = load_model_and_tokenizer(config.model.name, tokenizer_name=config.model.tokenizer_name,
                                                    torch_dtype=resolve_torch_dtype(config.model.dtype))
        if hasattr(model, "config"):
            model.config.use_cache = False
        _, probe = setup_probe(model, config.training.probe, config.training.lora,
                               checkpoint_path=Path(provenance["checkpoint_dir"]), seed=config.training.runtime.seed)
        layer = int(depth_dirs[0].name.split("_")[1])
        frames = {layer: score_split(probe, config, tokenizer, split=args.split, metadata=metadata, batch_size=1)}

    for depth in depth_dirs:
        layer = int(depth.name.split("_")[1])
        added = append(depth, args.split, frames[layer])
        print(f"{run_dir.name} layer {layer}: +{added} rollouts in {args.split}")


if __name__ == "__main__":
    main()
