"""Build a shuffled-frontier control variant of an existing dataset build.

The point of this variant is a permutation test: keep every rollout's actual
text, activations, and which rollouts count as positive exactly as they are,
but randomly reassign *which* frontier position each positive rollout is
labelled with, so a token's label no longer corresponds to anything real in
its own text. A probe (and any adapter) trained against these labels can
still fit them well -- optimization doesn't care whether a label is true --
but if it does, that is evidence the original frozen-vs-LoRA comparison this
control exists for is showing "supervised training can fit any label", not
"LoRA made degeneration more visible". See notebooks/lora_linearity_probe.py's
closing section for the full argument.

Every positive rollout in this corpus is exactly 4096 tokens long (only
rollouts that hit the token limit are ever sent to the judge, so a shuffled
onset_position from one positive rollout is always a valid position in any
other), which is confirmed here rather than assumed. Shuffling is a random
permutation of onset_position (and, for bookkeeping consistency, the
onset_metric that was paired with it) among positive rows, done separately
per split so no split boundary is crossed.

Nothing else about the build is duplicated: prompts, generations and labels
are symlinked from the source build, and the new dataset config points
activations_root straight at the source build's activation cache, since two
copies of ~36,300 rollouts of hidden states is not something to create by
accident.

    python scripts/build_shuffled_frontier_dataset.py \
        --source-build-root /capstor/.../degeneration-dataset-apertus-8b-instruct \
        --out-build-root /capstor/.../degeneration-dataset-apertus-8b-instruct-shuffled-frontier \
        --seed 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SYMLINKED_ARTIFACTS = ["prompts", "generations", "labels", "manifest.json"]


def build(source_build_root: Path, out_build_root: Path, *, seed: int) -> None:
    onset = pd.read_parquet(source_build_root / "onset_labels" / "onset_labels.parquet")

    positive = onset["is_positive"]
    bad_length = onset.loc[positive, "num_tokens"] != 4096
    if bad_length.any():
        raise ValueError(
            f"{int(bad_length.sum())} positive rollout(s) are not exactly 4096 tokens long, "
            "so a shuffled onset_position from another positive rollout is not guaranteed "
            "valid for them. Re-check this assumption before shuffling."
        )

    shuffled = onset.copy()
    rng = np.random.default_rng(seed)
    for split, group in onset[positive].groupby("split"):
        order = rng.permutation(group.index.to_numpy())
        shuffled.loc[group.index, "onset_position"] = onset.loc[order, "onset_position"].to_numpy()
        shuffled.loc[group.index, "onset_metric"] = onset.loc[order, "onset_metric"].to_numpy()
        unchanged = int((shuffled.loc[group.index, "onset_position"].to_numpy() == onset.loc[group.index, "onset_position"].to_numpy()).sum())
        print(f"split={split!r}: {len(group)} positive rows shuffled, {unchanged} landed on their own original value by chance")

    out_onset_dir = out_build_root / "onset_labels"
    out_onset_dir.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(out_onset_dir / "onset_labels.parquet", index=False)
    print(f"Wrote shuffled onset labels -> {out_onset_dir / 'onset_labels.parquet'}")

    for name in SYMLINKED_ARTIFACTS:
        target = source_build_root / name
        link = out_build_root / name
        if link.exists() or link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            raise FileExistsError(f"{link} already exists and does not point at {target}")
        os.symlink(target, link, target_is_directory=target.is_dir())
        print(f"Symlinked {link} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-build-root", type=Path, required=True)
    parser.add_argument("--out-build-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.source_build_root, args.out_build_root, seed=args.seed)


if __name__ == "__main__":
    main()
