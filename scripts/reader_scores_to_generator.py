"""Express a text-only reader's token scores in the generator's tokens.

A reader run is trained and scored on a re-tokenized build
(scripts/build_reader_dataset.py), so its scores sit on the reader's tokens.
The protocol has to be read against the generator's own onsets and token
counts, so that every scorer of a response is measured on the same positions.
Each generator token takes the score of the last reader token that had been
completed when the generator emitted it, which is the alignment the reader
build stored; a generator token before any completed reader token scores zero.

    python scripts/reader_scores_to_generator.py \\
        --scores-dir <run>/<attempt>/selected/layer_27/scores \\
        --reader-build-root <store>/degeneration-dataset-<gen>-reader-qwen2p5-1p5b \\
        --source-build-root <store>/degeneration-dataset-<gen> \\
        --out-dir <run>/<attempt>/selected_generator/layer_27/scores

Then evaluate the output directory's parent with scripts/evaluate_scores.py and
the source build's onset labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import build_scores, read_scores, write_scores  # noqa: E402


def load_alignment(reader_build_root: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, columns=["prompt_id", "rollout_idx", "generator_alignment"])
        for path in sorted((reader_build_root / "generations").glob("*/*.parquet"))
    ]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--reader-build-root", type=Path, required=True)
    parser.add_argument("--source-build-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    alignment = load_alignment(args.reader_build_root).set_index(["prompt_id", "rollout_idx"])["generator_alignment"]
    onset = pd.read_parquet(args.source_build_root / "onset_labels" / "onset_labels.parquet")
    onset = onset.set_index(["prompt_id", "rollout_idx"])

    for path in sorted(args.scores_dir.glob("*.parquet")):
        frame = read_scores(path, validate=False)
        records = []
        for row in frame.itertuples(index=False):
            key = (str(row.prompt_id), int(row.rollout_idx))
            align = np.asarray(alignment.loc[key], dtype=np.int64)
            reader_scores = np.asarray(row.scores, dtype=np.float32)
            # Reader positions past the scored length cannot occur for a
            # rollout scored in full; clip so a truncated one degrades safely.
            index = np.clip(align, -1, len(reader_scores) - 1)
            scores = np.where(index >= 0, reader_scores[np.maximum(index, 0)], 0.0).astype(np.float32)
            source = onset.loc[key]
            positive = bool(source["is_positive"]) and pd.notna(source["onset_position"])
            records.append({
                "prompt_id": key[0], "rollout_idx": key[1], "domain": str(source["domain"]),
                "split": str(source["split"]), "stop_reason": str(source["stop_reason"]),
                "num_tokens": int(source["num_tokens"]),
                "onset_position": float(source["onset_position"]) if pd.notna(source["onset_position"]) else None,
                "is_positive": positive, "scores": scores,
            })
        out = write_scores(build_scores(records), args.out_dir / path.name)
        print(f"{len(records):>6} rollouts -> {out}")


if __name__ == "__main__":
    main()
