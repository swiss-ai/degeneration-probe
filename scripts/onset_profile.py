"""Where a scorer's flags fall relative to the onset, at a fixed false-alarm budget.

Warning coverage over a window and pre-onset coverage before it summarise the
same quantity at two distances. This reports it at every distance: for each bin
of distance before the onset (and a few bins after it), the share of tokens of
degenerate responses that the scorer flags at its validation threshold, pooled
over responses and averaged over seeds. A scorer that reads the approaching
loop has a profile that rises toward the onset; one that flags degenerate
responses from their start has a flat one.

Two healthy references sit beside the profile. One is the share of all tokens
of healthy responses that the scorer flags, the level a degenerate response
would show if the scorer could not tell it apart. The other is that share by
absolute position in the healthy responses, which shows whether the scorer
flags more as a response grows. Few healthy responses reach the later
positions, so each position also carries the number of responses still running
there and a 95% interval from resampling responses, pooled over seeds.

    python scripts/onset_profile.py --scorer "apertus frozen L15" <depth dir> [...] \\
        --scorer ... --split test_indomain --budget 0.05 --out <csv>

Each depth directory holds scores/<split>.parquet and decision_thresholds.json;
several directories under one scorer are its seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import read_scores  # noqa: E402

BEFORE = [(lo, lo + 32) for lo in range(0, 1024, 32)]
AFTER = [(lo, lo + 32) for lo in range(0, 256, 32)] + [(256, 512), (512, 4096)]
POSITION_BIN, POSITION_END = 128, 2048


def profile(depth: Path, split: str, budget: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The profile rows of one seed, plus per healthy response its flags and tokens by position bin."""
    thresholds = json.loads((depth / "decision_thresholds.json").read_text())["thresholds"]
    tau = next(t["tau"] for t in thresholds if abs(t["target_negative_fpr"] - budget) < 1e-9)
    frame = read_scores(depth / "scores" / f"{split}.parquet", validate=False)
    # A fixed order, so the healthy responses line up across seeds when resampled.
    healthy = frame[~frame["is_positive"].astype(bool)].sort_values(["prompt_id", "rollout_idx"])
    frame = frame[frame["is_positive"].astype(bool)]
    edges = BEFORE + [(1024, None)]
    flagged = {("before", e): [0, 0] for e in edges}
    flagged.update({("after", e): [0, 0] for e in AFTER})
    for scores, onset in zip(frame["scores"], frame["onset_position"].astype(int)):
        above = np.asarray(scores, dtype=np.float64) >= tau
        for lo, hi in edges:
            start = 0 if hi is None else max(0, onset - hi)
            stop = max(0, onset - lo) if hi is not None else max(0, onset - 1024)
            span = above[start:stop]
            flagged[("before", (lo, hi))][0] += int(span.sum())
            flagged[("before", (lo, hi))][1] += span.size
        for lo, hi in AFTER:
            span = above[onset + lo : onset + hi]
            flagged[("after", (lo, hi))][0] += int(span.sum())
            flagged[("after", (lo, hi))][1] += span.size
    rows = []
    for (side, (lo, hi)), (hits, total) in flagged.items():
        label = f"{lo}-{hi}" if hi is not None else f">{lo}"
        rows.append({"side": side, "bin": label, "lo": lo, "flag_rate": hits / total if total else np.nan, "tokens": total})
    hits = total = 0
    bins = POSITION_END // POSITION_BIN
    position_hits, position_tokens = np.zeros((len(healthy), bins)), np.zeros((len(healthy), bins))
    for i, scores in enumerate(healthy["scores"]):
        above = np.asarray(scores, dtype=np.float64) >= tau
        hits, total = hits + int(above.sum()), total + above.size
        index = np.arange(min(above.size, POSITION_END)) // POSITION_BIN
        position_hits[i] = np.bincount(index, weights=above[: index.size], minlength=bins)
        position_tokens[i] = np.bincount(index, minlength=bins)
    rows.append({"side": "healthy", "bin": "all", "lo": 0, "flag_rate": hits / total if total else np.nan,
                 "tokens": total, "responses": len(healthy)})
    return pd.DataFrame(rows), position_hits, position_tokens


def position_rows(hits: list[np.ndarray], tokens: list[np.ndarray], draws: int = 1000) -> pd.DataFrame:
    """Share of healthy tokens flagged by position, averaged over seeds, with a response bootstrap."""
    hits, tokens = np.stack(hits), np.stack(tokens)
    if not all(np.array_equal(tokens[0], t) for t in tokens[1:]):
        raise ValueError("the seeds of one scorer were scored on different healthy responses")

    def rate(index: np.ndarray) -> np.ndarray:
        h, t = hits[:, index].sum(1), tokens[:, index].sum(1)
        return np.divide(h, t, out=np.full(h.shape, np.nan), where=t > 0).mean(0)

    responses = tokens.shape[1]
    rng = np.random.default_rng(0)
    boot = np.stack([rate(rng.integers(0, responses, responses)) for _ in range(draws)])
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    point = rate(np.arange(responses))
    return pd.DataFrame({
        "side": "healthy_position", "bin": [f"{lo}-{lo + POSITION_BIN}" for lo in range(0, POSITION_END, POSITION_BIN)],
        "lo": range(0, POSITION_END, POSITION_BIN), "flag_rate": point, "ci_low": low, "ci_high": high,
        "tokens": tokens[0].sum(0).astype(int), "responses": (tokens[0] > 0).sum(0), "seed": -1,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer", nargs="+", action="append", required=True, metavar=("NAME", "DIR"))
    parser.add_argument("--split", default="test_indomain")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for name, *dirs in args.scorer:
        hits, tokens = [], []
        for seed, depth in enumerate(dirs):
            part, position_hits, position_tokens = profile(Path(depth), args.split, args.budget)
            part["scorer"], part["seed"] = name, seed
            frames.append(part)
            hits.append(position_hits)
            tokens.append(position_tokens)
        frames.append(position_rows(hits, tokens).assign(scorer=name))
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = table.groupby(["scorer", "side", "lo", "bin"], sort=False)["flag_rate"].mean().reset_index()
    print(summary.pivot_table(index=["side", "lo", "bin"], columns="scorer", values="flag_rate", sort=False).round(3).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
