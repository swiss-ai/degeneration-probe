"""Turning the judge's onset quote into a token position.

The judge names where a rollout starts degenerating by quoting the text, which
is the form a human can check but not the form training needs. This module
locates that quote in the rollout's own token stream and reports the index of
its first token, counted from the start of the completion, which is the frame
every per-token target and every reported alarm position already uses.

The search is a binary search over decoded prefixes rather than a character
offset, because a character offset would have to assume that decoding tokens
one at a time and joining the pieces reproduces the text exactly, which is not
true in general. Decoding a prefix and asking whether the quote is inside it
makes no such assumption, and the result is verified against the located span
before it is returned.

Resolution is kept apart from labelling and written to its own table. It is the
one expensive step in the chain, it does not change unless the judge is re-run,
and every failure it hits is a fact worth keeping rather than a row to drop
silently: a quote the judge invented, or one that survives only in a form the
tokenizer cannot reproduce, should be visible as such.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.label import write_shard_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "dataset" / "builds" / "degeneration-dataset-apertus-8b-instruct.yaml"
)

QUOTE_POSITION_COLUMNS = [
    "prompt_id",
    "rollout_idx",
    "domain",
    "split",
    "stop_reason",
    "num_tokens",
    "judge_status",
    "is_degenerating",
    "onset_quote",
    "quote_words",
    "resolution",
    "onset_quote_position",
    "onset_quote_end",
    "onset_quote_tokens",
]

# Why a rollout has no usable position. Only "ok" yields one; the rest are kept
# so that a shrinking positive population can be attributed rather than guessed.
RESOLUTIONS = (
    "ok",
    "not_found",
    "empty_quote",
    "not_degenerating",
    "judge_failed",
    "unjudged",
)


def locate_quote(quote: str, token_ids: Sequence[int], tokenizer) -> Optional[slice]:
    """The span of tokens holding the quote's first occurrence, or None.

    First occurrence is the one that matters: when the quote is the unit the
    model has begun repeating it appears many times, and the onset is the
    earliest of them.
    """
    import torch

    from degeneration_probe.utils.tokenization import find_string_in_tokens

    if not quote:
        return None
    tokens = torch.as_tensor([int(t) for t in token_ids])
    if not len(tokens):
        return None
    try:
        return find_string_in_tokens(quote, tokens, tokenizer)
    except (AssertionError, ValueError, IndexError):
        return None


def load_judge_results(
    config: DatasetGenConfig, backends: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Every backend's verdicts, one row per rollout, best verdict winning.

    A rollout refused by one backend may have been answered by another, so the
    backends are pooled and a usable verdict is preferred over a failure.
    """
    directory = paths.llm_judge_dir(config)
    if backends is None:
        files = sorted(
            path
            for path in directory.glob("results_*.parquet")
            if path.suffixes == [".parquet"]
        )
    else:
        files = [paths.llm_judge_results_path(config, backend) for backend in backends]
    frames = []
    for path in files:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame["judge_backend"] = path.stem.replace("results_", "")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No judge results found under {directory}")
    pooled = pd.concat(frames, ignore_index=True)
    # A verdict with a quote beats one without, and any answer beats a refusal.
    # Ranked rather than merely sorted on usability, so that a backend which
    # refused a rollout cannot displace another backend's real verdict and turn
    # a known "not degenerating" into an unexplained failure.
    has_quote = pooled["onset_quote"].fillna("").str.len().gt(0)
    pooled["_rank"] = 2
    pooled.loc[pooled["status"] == "ok", "_rank"] = 1
    pooled.loc[(pooled["status"] == "ok") & has_quote, "_rank"] = 0
    pooled = pooled.sort_values("_rank", kind="stable")
    pooled = pooled.drop_duplicates(subset=["prompt_id", "rollout_idx"], keep="first")
    return pooled.drop(columns="_rank")


def _quote_of(row) -> str:
    """The row's quote as a string. A missing quote arrives as NaN, which is
    truthy, so it has to be tested for rather than trusted to be falsy."""
    quote = row.get("onset_quote")
    if quote is None or (isinstance(quote, float) and pd.isna(quote)):
        return ""
    return str(quote)


def _classify(row) -> str:
    if pd.isna(row.get("status")):
        return "unjudged"
    if row["status"] != "ok":
        return "judge_failed"
    if row.get("is_degenerating") is not True:
        return "not_degenerating"
    if not _quote_of(row):
        return "empty_quote"
    return "ok"


def resolve_domain_quote_positions(
    config: DatasetGenConfig,
    domain: str,
    judge: pd.DataFrame,
    tokenizer,
    prompt_id_to_domain: Dict[str, str],
    prompt_id_to_split: Dict[str, str],
) -> pd.DataFrame:
    """Locate every judged quote in one domain's rollouts.

    Only rollouts that hit the token cap are considered. A rollout that stopped
    on its own is not degenerating by construction, so a quote for it would be
    a contradiction rather than a label.
    """
    generations = pd.read_parquet(
        paths.generations_shard_path(config, domain, 0),
        columns=["prompt_id", "rollout_idx", "stop_reason", "num_tokens", "generated_token_ids"],
    )
    generations = generations[generations["stop_reason"] == "length"]
    merged = generations.merge(
        judge[["prompt_id", "rollout_idx", "status", "is_degenerating", "onset_quote"]],
        on=["prompt_id", "rollout_idx"],
        how="left",
    )

    records: List[dict] = []
    for row in merged.to_dict("records"):
        resolution = _classify(row)
        quote = _quote_of(row)
        start = end = None
        if resolution == "ok":
            span = locate_quote(quote, row["generated_token_ids"], tokenizer)
            if span is None:
                resolution = "not_found"
            else:
                start, end = int(span.start), int(span.stop)

        records.append(
            {
                "prompt_id": row["prompt_id"],
                "rollout_idx": int(row["rollout_idx"]),
                "domain": prompt_id_to_domain.get(row["prompt_id"], domain),
                "split": prompt_id_to_split.get(row["prompt_id"]),
                "stop_reason": row["stop_reason"],
                "num_tokens": int(row["num_tokens"]),
                "judge_status": row.get("status"),
                "is_degenerating": row.get("is_degenerating"),
                "onset_quote": quote or None,
                "quote_words": len(quote.split()) if quote else 0,
                "resolution": resolution,
                "onset_quote_position": start,
                "onset_quote_end": end,
                "onset_quote_tokens": (end - start) if start is not None else None,
            }
        )
    return pd.DataFrame.from_records(records, columns=QUOTE_POSITION_COLUMNS)


def resolve_quote_positions(
    config: DatasetGenConfig,
    domains: Optional[Sequence[str]] = None,
    backends: Optional[Sequence[str]] = None,
    tokenizer=None,
) -> pd.DataFrame:
    from transformers import AutoTokenizer

    domains = domains or sorted(paths.configured_domain_names(config))
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.model_name)

    from degeneration_probe.dataset_gen.onset_labels import _load_prompt_id_to_split

    prompts = pd.read_parquet(paths.prompts_path(config), columns=["prompt_id", "domain"])
    prompt_id_to_domain = dict(zip(prompts["prompt_id"], prompts["domain"]))
    prompt_id_to_split = _load_prompt_id_to_split(config)
    judge = load_judge_results(config, backends)

    frames = [
        resolve_domain_quote_positions(
            config, domain, judge, tokenizer, prompt_id_to_domain, prompt_id_to_split
        )
        for domain in domains
    ]
    return pd.concat(frames, ignore_index=True)


def report(frame: pd.DataFrame) -> str:
    lines = [
        f"Capped rollouts: {len(frame)}",
        "",
        "Resolution:",
    ]
    counts = frame["resolution"].value_counts()
    for name in RESOLUTIONS:
        if name in counts:
            lines.append(f"  {name:<18} {counts[name]:>5}  ({counts[name] / len(frame):.1%})")
    resolved = frame[frame["resolution"] == "ok"]
    lines += ["", "Positions resolved, by split:"]
    by_split = frame.groupby("split", dropna=False).apply(
        lambda part: pd.Series(
            {
                "capped": len(part),
                "resolved": int((part["resolution"] == "ok").sum()),
            }
        ),
        include_groups=False,
    )
    lines.append(by_split.to_string())
    lines += ["", "By domain:"]
    by_domain = frame.groupby("domain", dropna=False).apply(
        lambda part: pd.Series(
            {
                "capped": len(part),
                "resolved": int((part["resolution"] == "ok").sum()),
                "lost": int((part["resolution"] != "ok").sum()),
            }
        ),
        include_groups=False,
    )
    lines.append(by_domain.to_string())
    if len(resolved):
        fraction = resolved["onset_quote_position"] / resolved["num_tokens"]
        lines += [
            "",
            f"Onset position (tokens): median {resolved['onset_quote_position'].median():.0f}, "
            f"mean {resolved['onset_quote_position'].mean():.0f}",
            f"Onset as a fraction of the rollout: median {fraction.median():.2f}",
            f"Quote span (tokens): median {resolved['onset_quote_tokens'].median():.0f}",
            f"Quote length (words): median {resolved['quote_words'].median():.0f}",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Judge backends to pool (default: every results_*.parquet present).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Resolve only the first N capped rollouts per domain, for a quick check.",
    )
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)
    frame = resolve_quote_positions(config, domains=args.domains, backends=args.backends)
    if args.limit is not None:
        frame = frame.groupby("domain", group_keys=False).head(args.limit)

    out_path = paths.onset_quote_positions_path(config)
    write_shard_atomic(frame, out_path)
    print(f"Wrote {len(frame)} rows to {out_path}\n")
    print(report(frame))


if __name__ == "__main__":
    main()
