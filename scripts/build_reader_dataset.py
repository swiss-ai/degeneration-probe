"""Re-tokenize a dataset build for a text-only reader model.

The text-only baseline is a separate, smaller language model that reads the
prompt and the generated text, never the generator's hidden states, and is
trained with the probes' own recipe to score every token. It has its own
tokenizer, so the generations, the onset positions and the token counts of the
source build are re-expressed in its tokens, and the training and scoring code
then runs on the derived build unchanged.

Positions are mapped through characters. Each generator token is decoded
incrementally, which gives the exact character span it adds to the response,
and the reader's tokenizer reports the span of each of its tokens on the same
text. The reader onset is the reader token that contains the first character of
the generator's onset token. For scoring, each generator token is assigned the
last reader token that ends at or before the end of that generator token, so a
score read back in generator units never uses text the generator had not yet
produced (``-1`` when no reader token has completed yet).

    python scripts/build_reader_dataset.py \\
        --source-build-root <store>/degeneration-dataset-apertus-8b-instruct \\
        --generator-tokenizer swiss-ai/Apertus-8B-Instruct-2509 \\
        --reader-tokenizer Qwen/Qwen2.5-1.5B \\
        --out-build-root <store>/degeneration-dataset-apertus-8b-instruct-reader-qwen2p5-1p5b

Prompts, labels, splits and the manifest are symlinked from the source build.
The generations shards carry the reader ids and, per rollout, the alignment
back to generator tokens.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SYMLINKED_ARTIFACTS = ["prompts", "labels", "splits", "manifest.json"]
_TOKENIZERS = {}


def _tokenizers(generator: str, reader: str):
    if not _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS["generator"] = AutoTokenizer.from_pretrained(generator)
        _TOKENIZERS["reader"] = AutoTokenizer.from_pretrained(reader)
    return _TOKENIZERS["generator"], _TOKENIZERS["reader"]


def generator_char_ends(tokenizer, token_ids) -> tuple[str, np.ndarray]:
    """The decoded response and the character offset at which each token ends."""
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(skip_special_tokens=True)
    pieces, ends, length = [], np.zeros(len(token_ids), dtype=np.int64), 0
    try:
        for index, token in enumerate(token_ids):
            chunk = stream.step(tokenizer._tokenizer, int(token))
            if chunk:
                pieces.append(chunk)
                length += len(chunk)
            ends[index] = length
        return "".join(pieces), ends
    except Exception:
        return prefix_char_ends(tokenizer, token_ids)


def prefix_char_ends(tokenizer, token_ids) -> tuple[str, np.ndarray]:
    """The same offsets from decoding every prefix, for streams the incremental decoder rejects.

    A prefix that ends inside a multi-byte character decodes with a trailing
    replacement character, which is dropped, and the offsets are made
    non-decreasing, so a character is credited to the token that completes it.
    """
    ids = [int(t) for t in token_ids]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    ends = np.zeros(len(ids), dtype=np.int64)
    for index in range(len(ids)):
        prefix = tokenizer.decode(ids[: index + 1], skip_special_tokens=True).rstrip("\ufffd")
        ends[index] = min(len(prefix), len(text))
    return text, np.maximum.accumulate(ends)


def convert_shard(task) -> dict:
    path, onset_rows, generator, reader = task
    gen_tok, read_tok = _tokenizers(generator, reader)
    shard = pd.read_parquet(path, columns=["prompt_id", "rollout_idx", "generated_token_ids"])
    shard = shard.merge(onset_rows, on=["prompt_id", "rollout_idx"], how="inner", validate="one_to_one")
    rows, onsets = [], []
    for row in shard.itertuples(index=False):
        text, char_ends = generator_char_ends(gen_tok, row.generated_token_ids)
        encoded = read_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        reader_ends = np.asarray([end for _, end in encoded["offset_mapping"]], dtype=np.int64)
        alignment = np.searchsorted(reader_ends, char_ends, side="right") - 1
        onset = None
        if pd.notna(row.onset_position):
            onset_token = int(row.onset_position)
            onset_char = int(char_ends[onset_token - 1]) if onset_token > 0 else 0
            onset = int(np.searchsorted(reader_ends, onset_char, side="right"))
        rows.append({
            "prompt_id": row.prompt_id,
            "rollout_idx": int(row.rollout_idx),
            "generated_token_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
            "generator_alignment": alignment.astype(np.int32),
            "generator_num_tokens": int(len(row.generated_token_ids)),
        })
        onsets.append({"prompt_id": row.prompt_id, "rollout_idx": int(row.rollout_idx),
                       "reader_onset_position": onset, "reader_num_tokens": len(encoded["input_ids"])})
    return {"path": path, "rows": rows, "onsets": onsets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-build-root", type=Path, required=True)
    parser.add_argument("--out-build-root", type=Path, required=True)
    parser.add_argument("--generator-tokenizer", required=True)
    parser.add_argument("--reader-tokenizer", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()

    source, out = args.source_build_root, args.out_build_root
    out.mkdir(parents=True, exist_ok=True)
    for name in SYMLINKED_ARTIFACTS:
        if (source / name).exists() and not (out / name).exists():
            os.symlink(source / name, out / name)

    onset = pd.read_parquet(source / "onset_labels" / "onset_labels.parquet")
    keys = onset[["prompt_id", "rollout_idx", "onset_position"]]
    tasks = []
    for domain in sorted(onset["domain"].astype(str).unique()):
        domain_keys = keys[onset["domain"].astype(str) == domain]
        for path in sorted((source / "generations" / domain).glob("*.parquet")):
            tasks.append((path, domain_keys, args.generator_tokenizer, args.reader_tokenizer))

    mapped = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(convert_shard, tasks):
            target = out / "generations" / result["path"].parent.name / result["path"].name
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["rows"]).to_parquet(target, index=False)
            mapped += result["onsets"]
            print(f"{len(result['rows']):>5} rollouts -> {target}", flush=True)

    mapped = pd.DataFrame(mapped)
    reader_onset = onset.merge(mapped, on=["prompt_id", "rollout_idx"], how="left", validate="one_to_one")
    missing = reader_onset["reader_num_tokens"].isna()
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} rollouts have no generation shard")
    reader_onset["generator_onset_position"] = reader_onset["onset_position"]
    reader_onset["generator_num_tokens"] = reader_onset["num_tokens"]
    reader_onset["onset_position"] = reader_onset["reader_onset_position"]
    reader_onset["num_tokens"] = reader_onset["reader_num_tokens"].astype(int)
    reader_onset = reader_onset.drop(columns=["reader_onset_position", "reader_num_tokens"])
    (out / "onset_labels").mkdir(exist_ok=True)
    reader_onset.to_parquet(out / "onset_labels" / "onset_labels.parquet", index=False)
    ratio = reader_onset["num_tokens"] / reader_onset["generator_num_tokens"]
    print(f"{len(reader_onset)} rollouts; reader/generator token ratio median {ratio.median():.3f}, "
          f"max reader length {reader_onset['num_tokens'].max()}")


if __name__ == "__main__":
    main()
