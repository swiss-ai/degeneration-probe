"""Build a rollout-label-shuffled control variant of an existing dataset build.

The first control (build_shuffled_frontier_dataset.py) only scrambles *where*
the frontier falls within an already-positive rollout. It leaves rollout-level
positive/negative status untouched, so with horizon=1024 most positives still
get most of their tokens labelled 1 regardless of the shuffled position
(frontier_hard_targets: start = max(onset - horizon, 0)) -- the adapter can
still learn "this whole rollout is one of the known positives" and that
control cannot tell that apart from a genuine content signal. See
notebooks/lora_linearity_probe.py's closing section and the review comment
this script responds to.

This control instead scrambles *which rollouts are positive at all*: a random
subset of real healthy (stop_reason == "eos") rollouts, equal in count to the
split's real positives, is promoted to "positive" with a synthetic frontier;
the real positives are demoted to unused. A promoted rollout's own text never
actually degenerates, so if training against these labels still produces the
same kind of clean separation the real run shows, that is evidence of fitting
an arbitrary label assignment, not of reading real degenerate content.

Why this can't reuse real onset_position values the way the first control
does: real onsets run past 3000 in some rollouts, but eos rollouts (which
stopped on their own) are almost all much shorter -- the 90th percentile is
only ~1000 tokens per split, and only a handful of eos rows per split are
even long enough to host the largest real onsets. There is no length-matched
supply to permute onset values onto. So a promoted rollout instead gets a
uniformly random cut point within its own length -- an arbitrary decorrelated
label, which is exactly what a "does detection work under a fake label"
control needs; it does not attempt to imitate the real onset distribution
the way the first control's shuffle does, and should not be read as doing so.

Every other artifact (prompts, generations, labels, activations) is symlinked
or pointed at the source build unchanged, same as the first control.

    python scripts/build_shuffled_rollout_label_dataset.py \
        --source-build-root /capstor/.../degeneration-dataset-apertus-8b-instruct \
        --out-build-root /capstor/.../degeneration-dataset-apertus-8b-instruct-shuffled-rollout-label \
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
    shuffled = onset.copy()
    rng = np.random.default_rng(seed)

    for split, split_index in onset.groupby("split").groups.items():
        split_rows = onset.loc[split_index]
        positive_index = split_rows.index[split_rows["is_positive"]]
        eos_index = split_rows.index[split_rows["stop_reason"] == "eos"]
        n_needed = len(positive_index)
        if n_needed == 0:
            continue
        if len(eos_index) < n_needed:
            raise ValueError(
                f"split {split!r} has only {len(eos_index)} eos rollouts but needs "
                f"{n_needed} to promote -- cannot build a same-size fake-positive set"
            )

        promoted = rng.choice(eos_index.to_numpy(), size=n_needed, replace=False)
        num_tokens = shuffled.loc[promoted, "num_tokens"].to_numpy()
        fake_onset = rng.integers(0, num_tokens)  # uniform in [0, num_tokens), always valid

        shuffled.loc[promoted, "is_positive"] = True
        shuffled.loc[promoted, "stop_reason"] = "length"  # bypass the eos short-circuit in frontier_hard_targets
        shuffled.loc[promoted, "onset_position"] = fake_onset.astype(float)
        shuffled.loc[promoted, "onset_resolution"] = "ok"
        shuffled.loc[promoted, "onset_metric"] = np.nan  # arbitrary cut point, no real repetition score to report

        shuffled.loc[positive_index, "is_positive"] = False  # demoted: excluded from training, same as any
        # other capped-but-unflagged rollout (stop_reason stays "length", is_positive False -> unused)

        print(
            f"split={split!r}: promoted {n_needed} eos rollouts to fake-positive, "
            f"demoted {n_needed} real positives to unused"
        )

    out_onset_dir = out_build_root / "onset_labels"
    out_onset_dir.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(out_onset_dir / "onset_labels.parquet", index=False)
    print(f"Wrote rollout-label-shuffled onset labels -> {out_onset_dir / 'onset_labels.parquet'}")

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
