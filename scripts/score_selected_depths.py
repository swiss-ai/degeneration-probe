"""Score every depth of a run at the checkpoint selected for it, in one pass per split.

A run that trained a head per depth saves every checkpoint as one directory per
depth, each an ordinary single-layer probe. Selection
(``scripts/select_checkpoints.py``) names a different step for each depth, so no
single checkpoint holds the heads to score. This assembles one: a composite
checkpoint directory whose ``layer_NN`` entries link to the selected step of each
depth. The multi-depth probe loads it like any other checkpoint, and the split is
read once for all depths instead of once per depth.

Scores land in ``<run_dir>/<subdir>/layer_NN/scores/<split>.parquet`` with a
provenance file naming the step behind each depth, so every directory is an
ordinary scores directory for ``scripts/evaluate_scores.py``.

    python scripts/score_selected_depths.py --run-dir outputs/<run>/<attempt>
    python scripts/score_selected_depths.py --run-dir ... --splits val --layers 8 15 --subdir smoke

Only the cached regime is supported: the point is the shared read of the
activation cache, and a frozen run reads the same representation from it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import write_scores
from degeneration_probe.probes.linear_probe import setup_cached_probe
from scripts.score_rollouts import (
    SCORES_DIR,
    check_provenance,
    load_run_config,
    rollout_metadata,
    score_split_by_layer,
    write_provenance,
)

SELECTION_FILE = "checkpoint_selection.json"
DEFAULT_SPLITS = ["val", "test_indomain", "test_heldout_domains"]


def read_selection(path: Path) -> Dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["layer"]): int(row["selected_step"]) for row in payload["depths"]}


def assemble_composite(run_dir: Path, steps_by_layer: Dict[int, int], name: str) -> Path:
    """One checkpoint directory whose depths point at their selected steps.

    Relative links, so the run directory can move as a whole. ``probe_config.json``
    is copied from the first depth's own checkpoint: the scorer only checks that
    the file exists before touching the GPU, and every head carries its own.
    """
    composite = run_dir / name
    composite.mkdir(exist_ok=True)
    for layer, step in sorted(steps_by_layer.items()):
        source = run_dir / f"checkpoint-{step}" / f"layer_{layer:02d}"
        if not (source / "probe_config.json").is_file():
            raise FileNotFoundError(f"no head for layer {layer} at {source}")
        link = composite / f"layer_{layer:02d}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(source, composite), target_is_directory=True)
    marker = composite / "probe_config.json"
    first = min(steps_by_layer)
    marker.write_text(
        (run_dir / f"checkpoint-{steps_by_layer[first]}" / f"layer_{first:02d}" / "probe_config.json").read_text(),
        encoding="utf-8",
    )
    (composite / "composition.json").write_text(
        json.dumps({f"layer_{l:02d}": f"checkpoint-{s}" for l, s in sorted(steps_by_layer.items())}, indent=2),
        encoding="utf-8",
    )
    return composite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="A run directory (an attempt).")
    parser.add_argument("--selection", type=Path, default=None, help=f"default <run_dir>/{SELECTION_FILE}")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="A subset of depths (default: all selected).")
    parser.add_argument("--subdir", default="selected", help="Where under the run the per-depth score directories go.")
    parser.add_argument("--composite-name", default="checkpoint-selected")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_run_config(run_dir)
    if config.training.features.regime != "cached":
        raise SystemExit("only cached-regime runs can share one pass over the activation cache")
    if config.training.probe.layers is None:
        raise SystemExit("this run trained a single depth; use scripts/score_rollouts.py")

    steps_by_layer = read_selection(args.selection or run_dir / SELECTION_FILE)
    if args.layers:
        missing = set(args.layers) - set(steps_by_layer)
        if missing:
            raise SystemExit(f"no selection for layers {sorted(missing)}")
        steps_by_layer = {l: steps_by_layer[l] for l in args.layers}
    layers = sorted(steps_by_layer)
    config.training.probe.layers = layers

    out_root = run_dir / args.subdir
    for layer in layers:
        check_provenance(out_root / f"layer_{layer:02d}", f"checkpoint-{steps_by_layer[layer]}", layer)

    composite = assemble_composite(run_dir, steps_by_layer, args.composite_name)
    print(f"Composite checkpoint {composite}: {len(layers)} depths, steps {sorted(set(steps_by_layer.values()))}")

    hidden_size = int(list(pd.read_parquet(config.dataset.activations_manifest_path, columns=["shape"])["shape"].iloc[0])[-1])
    config.dataset.sampling.evaluation_negative_rollouts_per_positive = None
    probe = setup_cached_probe(
        config.training.probe,
        hidden_size=hidden_size,
        checkpoint_path=composite,
        seed=config.training.runtime.seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    metadata = rollout_metadata(config)

    for split in args.splits:
        frames = score_split_by_layer(
            probe, config, None, split=split, metadata=metadata, batch_size=args.batch_size, layers=layers
        )
        for layer, frame in sorted(frames.items()):
            target = out_root / f"layer_{layer:02d}"
            target.mkdir(parents=True, exist_ok=True)
            path = write_scores(frame, target / SCORES_DIR / f"{split}.parquet")
            write_provenance(
                target,
                run_dir=run_dir,
                checkpoint=f"checkpoint-{steps_by_layer[layer]}",
                checkpoint_dir=run_dir / f"checkpoint-{steps_by_layer[layer]}" / f"layer_{layer:02d}",
                layer=layer,
                regime=config.training.features.regime,
                splits=[split],
            )
        sample = next(iter(frames.values()))
        print(
            f"  {split:<22} {len(sample):>6} rollouts ({int(sample['is_positive'].sum())} positive), "
            f"{int(sample['num_tokens'].sum()):>10,} tokens x {len(frames)} depths -> {out_root}/layer_NN/{SCORES_DIR}/{split}.parquet"
        )


if __name__ == "__main__":
    main()
