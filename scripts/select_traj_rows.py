"""Select rows from the degeneration-probe train arrow for trajectory plots.

Picks 9 rows with degenerating=True, diversified across source_dataset and
completion length, each containing >=30 degenerate tokens. Plus 1 row with
degenerating=False and completion_len >=100 as a clean control.

Prints the chosen row indices to stdout, one per line, in the order:
9 degenerating rows then 1 clean row. Also prints a short summary on stderr.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_ARROW = (
    "/capstor/scratch/cscs/lbaggi/feature-probes/hf_cache/datasets/"
    "lorenzo0312___degeneration-probe-instruct-token-level-balanced/default/0.0.0/"
    "55831e27b7609554a4f3d40a3a6f18e481f24d4c/"
    "degeneration-probe-instruct-token-level-balanced-train.arrow"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow", default=DEFAULT_ARROW)
    parser.add_argument("--n-degen", type=int, default=9)
    parser.add_argument("--min-degen-tokens", type=int, default=30)
    parser.add_argument("--min-clean-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=20000,
        help="Only inspect the first N rows of the arrow file (for speed).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with pa.memory_map(args.arrow, "r") as source:
        table = ipc.open_stream(source).read_all()

    n_rows = min(table.num_rows, args.max_scan)
    print(f"scanning first {n_rows} of {table.num_rows} rows", file=sys.stderr)

    degen_by_source: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    clean_candidates: list[tuple[int, int, str]] = []

    sources_col = table.column("source_dataset").to_pylist()[:n_rows]
    degen_col = table.column("degenerating").to_pylist()[:n_rows]
    chunks_col = table.column("chunk_summary").to_pylist()[:n_rows]

    for i in range(n_rows):
        chunks = chunks_col[i]
        n_tok = len(chunks)
        n_degen_tok = sum(int(c["degenerating"]) for c in chunks)
        src = sources_col[i]
        is_degen = bool(degen_col[i])
        if is_degen and n_degen_tok >= args.min_degen_tokens:
            degen_by_source[src].append((i, n_tok, n_degen_tok))
        elif not is_degen and n_tok >= args.min_clean_len and n_degen_tok == 0:
            clean_candidates.append((i, n_tok, src))

    print(
        "degenerating pool by source: "
        + ", ".join(f"{s}={len(v)}" for s, v in degen_by_source.items()),
        file=sys.stderr,
    )
    print(f"clean candidates: {len(clean_candidates)}", file=sys.stderr)

    if 612 in [c[0] for c in degen_by_source.get("zwhe99/DeepMath-103K", [])]:
        pass  # keep row 612 unconditionally as one of the picks

    selected_degen: list[tuple[int, int, int, str]] = []
    # round-robin across sources until we have n_degen picks
    sources = sorted(degen_by_source.keys())
    pools = {s: list(degen_by_source[s]) for s in sources}
    # sort each pool by completion_len descending so we prefer informative rows
    for s in sources:
        pools[s].sort(key=lambda t: t[1], reverse=True)

    forced = []
    for s in sources:
        for j, (idx, n_tok, n_degen_tok) in enumerate(pools[s]):
            if idx == 612:
                forced.append((idx, n_tok, n_degen_tok, s))
                pools[s].pop(j)
                break

    for idx, n_tok, n_degen_tok, s in forced:
        selected_degen.append((idx, n_tok, n_degen_tok, s))

    # pick varied lengths: split each source's pool into 3 length buckets and pick
    # from different buckets in round-robin
    src_buckets: dict[str, list[list[tuple[int, int, int]]]] = {}
    for s in sources:
        pool = pools[s]
        if not pool:
            src_buckets[s] = [[], [], []]
            continue
        lens = sorted({t[1] for t in pool})
        if len(lens) >= 3:
            q1 = np.quantile([t[1] for t in pool], 1 / 3)
            q2 = np.quantile([t[1] for t in pool], 2 / 3)
            short = [t for t in pool if t[1] <= q1]
            mid = [t for t in pool if q1 < t[1] <= q2]
            long_ = [t for t in pool if t[1] > q2]
        else:
            short, mid, long_ = pool, [], []
        for bucket in (short, mid, long_):
            rng.shuffle(bucket)
        src_buckets[s] = [short, mid, long_]

    bucket_idx = 0
    while len(selected_degen) < args.n_degen:
        progress = False
        for s in sources:
            if len(selected_degen) >= args.n_degen:
                break
            bucket = src_buckets[s][bucket_idx % 3]
            if not bucket:
                continue
            cand = bucket.pop()
            if cand[0] in {x[0] for x in selected_degen}:
                continue
            selected_degen.append((cand[0], cand[1], cand[2], s))
            progress = True
        bucket_idx += 1
        if not progress and bucket_idx > 10:
            break

    # clean: prefer one with mid-length completion
    clean_candidates.sort(key=lambda t: abs(t[1] - 300))
    chosen_clean = clean_candidates[0] if clean_candidates else None

    print("=== SELECTED DEGENERATING ROWS ===", file=sys.stderr)
    for idx, n_tok, n_degen_tok, s in selected_degen:
        print(
            f"  row={idx:6d}  src={s:30s}  n_tok={n_tok:4d}  n_degen={n_degen_tok:4d}",
            file=sys.stderr,
        )
    print("=== SELECTED CLEAN ROW ===", file=sys.stderr)
    if chosen_clean is not None:
        idx, n_tok, s = chosen_clean
        print(f"  row={idx:6d}  src={s:30s}  n_tok={n_tok:4d}", file=sys.stderr)
    else:
        print("  none found", file=sys.stderr)

    for idx, *_ in selected_degen:
        print(idx)
    if chosen_clean is not None:
        print(chosen_clean[0])


if __name__ == "__main__":
    main()
