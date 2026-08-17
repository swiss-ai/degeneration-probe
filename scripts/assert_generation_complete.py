"""Refuse to run a downstream stage against a half-generated build.

Generation runs as a fixed-length chain of wall-clock-limited waves per domain,
so it can terminate having produced only part of a domain. The stages that read
it do not all fail safely on partial input: labeling rewrites the per-prompt
statistics wholesale from whatever it is given, and the judge's calibration
sample is written once and then reused for every later run, so either one will
quietly bake in a partial build and keep serving it as if it were the real one.

This is the guard those stages run first. It compares the rollouts actually on
disk against the number the config asks for, per domain, and exits non-zero
with a per-domain breakdown if any are short -- which, submitted as a job
dependency, turns "silently wrong results" into "the next stage refused to
start, and the log says which domains are missing how many rollouts."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def expected_rollouts_per_domain(config: DatasetGenConfig) -> dict[str, int]:
    """How many rollouts each domain should hold, from the prompt pool itself.

    Read off the written pool rather than the configured ``n_prompts`` so that a
    source clamped to fewer rows than requested (aime_2025 has only 30) counts
    as complete at what it could actually supply.
    """
    prompts = pd.read_parquet(paths.prompts_path(config), columns=["prompt_id", "domain"])
    counts = prompts.groupby("domain").size()
    return {domain: int(n) * config.n_rollouts_per_prompt for domain, n in counts.items()}


def actual_rollouts_per_domain(config: DatasetGenConfig) -> dict[str, int]:
    actual: dict[str, int] = {}
    for domain in sorted(paths.configured_domain_names(config)):
        shard = paths.generations_shard_path(config, domain, 0)
        if not shard.exists():
            actual[domain] = 0
            continue
        actual[domain] = len(pd.read_parquet(shard, columns=["prompt_id"]))
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)
    expected = expected_rollouts_per_domain(config)
    actual = actual_rollouts_per_domain(config)

    short = {d: (actual.get(d, 0), n) for d, n in expected.items() if actual.get(d, 0) < n}

    print(f"Generation completeness for {args.config}")
    for domain in sorted(expected):
        got, want = actual.get(domain, 0), expected[domain]
        mark = "ok " if got >= want else "SHORT"
        print(f"  {mark} {domain:24s} {got:6d} / {want:6d}")
    print(f"  total {sum(actual.values())} / {sum(expected.values())}")

    if short:
        print(
            f"\nINCOMPLETE: {len(short)} domain(s) short by "
            f"{sum(w - g for g, w in short.values())} rollout(s). "
            "Resubmit generation for those domains before running this stage.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nComplete.")


if __name__ == "__main__":
    main()
