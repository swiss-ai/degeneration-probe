"""Which tokens of a rollout a training run actually learns from.

The five rules form a ladder rather than a menu: each rung changes exactly one
decision relative to the one before it, so comparing adjacent rungs isolates
that decision instead of confounding several. Every rung draws from the same
pool, the entire rollout including the region past the frontier, because
keeping the pool fixed is what makes an adjacent-rung difference attributable
to the selection rule rather than to the data.

    all_tokens          every token, tiled into windows, no subsampling
    rollout_balanced    a fixed budget per rollout, drawn independently
    random_window       the same budget, now contiguous
    frontier_window     the same window, anchored on the frontier
    ... + hard negatives  negative windows biased toward repetitive spans

Rungs 2 to 5 all keep a fixed budget per rollout, so token count per rollout
stops being a confound after rung 1. A rollout shorter than the budget
contributes every token it has: falling short is a property of the rollout, not
a reason to pad or drop it.

The output of every rule is the same descriptor, a rollout and the token
positions selected from it, so the rest of the pipeline never learns which rule
produced a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

STRATEGIES = {
    "all_tokens",
    "rollout_balanced",
    "random_window",
    "frontier_window",
    "frontier_window_hard_negative",
}
ANCHORS = {"trailing", "centered"}


@dataclass(frozen=True)
class Window:
    """Token positions selected from one rollout."""

    record_index: int
    positions: np.ndarray

    def __len__(self) -> int:
        return int(self.positions.size)


def _tile(record_index: int, num_tokens: int, window_size: int) -> List[Window]:
    return [
        Window(record_index, np.arange(start, min(start + window_size, num_tokens)))
        for start in range(0, num_tokens, window_size)
    ]


def _contiguous(record_index: int, start: int, num_tokens: int, window_size: int) -> Window:
    start = int(np.clip(start, 0, max(num_tokens - window_size, 0)))
    return Window(record_index, np.arange(start, min(start + window_size, num_tokens)))


def _random_placement(
    record_index: int, num_tokens: int, window_size: int, rng: np.random.Generator
) -> Window:
    if num_tokens <= window_size:
        return Window(record_index, np.arange(num_tokens))
    return _contiguous(record_index, rng.integers(0, num_tokens - window_size + 1), num_tokens, window_size)


def _frontier_placement(
    record_index: int, num_tokens: int, onset: int, window_size: int, anchor: str
) -> Window:
    if anchor == "trailing":
        # The window ends at the frontier: exactly the context a live decision
        # would have, with nothing after "now".
        start = onset - window_size
    else:
        start = onset - window_size // 2
    return _contiguous(record_index, start, num_tokens, window_size)


def _weighted_placement(
    record_index: int,
    num_tokens: int,
    window_size: int,
    hardness: np.ndarray,
    rng: np.random.Generator,
) -> Window:
    """Place a window where a rollout looks most repetitive."""
    if num_tokens <= window_size:
        return Window(record_index, np.arange(num_tokens))
    starts = np.arange(num_tokens - window_size + 1)
    scores = np.asarray(hardness, dtype=np.float64)[: num_tokens]
    if scores.size < num_tokens:
        scores = np.pad(scores, (0, num_tokens - scores.size))
    scores = np.nan_to_num(scores, nan=0.0)
    # Total repetitiveness inside each candidate window, via a prefix sum.
    cumulative = np.concatenate([[0.0], np.cumsum(scores)])
    weights = cumulative[starts + window_size] - cumulative[starts]
    if weights.sum() <= 0:
        return _random_placement(record_index, num_tokens, window_size, rng)
    return _contiguous(
        record_index, rng.choice(starts, p=weights / weights.sum()), num_tokens, window_size
    )


def build_windows(
    records: Sequence,
    *,
    strategy: str,
    window_size: int,
    rng: np.random.Generator,
    anchor: str = "trailing",
    hard_negative_fraction: float = 0.5,
    hardness: Optional[Dict[int, np.ndarray]] = None,
) -> List[Window]:
    """Select the tokens one epoch of training will see."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown sampling strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor {anchor!r}; expected one of {sorted(ANCHORS)}")
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")

    windows: List[Window] = []
    for index, record in enumerate(records):
        num_tokens = len(record.targets)
        if num_tokens == 0:
            continue

        if strategy == "all_tokens":
            windows.extend(_tile(index, num_tokens, window_size))
            continue

        if strategy == "rollout_balanced":
            # A fixed budget, but drawn independently rather than contiguously:
            # this rung asks whether correcting the population helps at all,
            # before contiguity is introduced.
            size = min(window_size, num_tokens)
            positions = np.sort(rng.choice(num_tokens, size=size, replace=False))
            windows.append(Window(index, positions))
            continue

        if strategy == "random_window":
            windows.append(_random_placement(index, num_tokens, window_size, rng))
            continue

        # Both frontier rungs anchor positives on the frontier and differ only
        # in how they place the negatives.
        if record.is_positive and record.onset_position is not None:
            windows.append(
                _frontier_placement(index, num_tokens, int(record.onset_position), window_size, anchor)
            )
        elif strategy == "frontier_window_hard_negative" and rng.random() < hard_negative_fraction:
            if hardness is None or index not in hardness:
                raise ValueError(
                    "frontier_window_hard_negative needs a per-token repetition signal for "
                    f"every negative rollout; none was supplied for record {index}"
                )
            windows.append(_weighted_placement(index, num_tokens, window_size, hardness[index], rng))
        else:
            windows.append(_random_placement(index, num_tokens, window_size, rng))
    return windows


def compose_batches(
    windows: Sequence[Window],
    records: Sequence,
    *,
    batch_size: int,
    positive_fraction: float,
    rng: np.random.Generator,
    max_rollouts_per_prompt: Optional[int] = None,
) -> List[List[int]]:
    """Group windows into batches with a fixed class mix and domain spread.

    Shuffling a flat list gives batches that are almost entirely negative, so
    many steps carry no positive gradient at all and the loss swings violently
    under a class weight. Composing instead fixes the mix per batch, draws
    negatives in proportion to each domain's share rather than uniformly over
    the pool, and caps how many rollouts of one prompt are used, since a prompt
    whose rollouts all degenerate would otherwise flood training with near
    duplicates.
    """
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError(f"positive_fraction must lie strictly between 0 and 1, got {positive_fraction}")

    kept = _cap_per_prompt(windows, records, max_rollouts_per_prompt, rng)
    positives = [i for i in kept if records[windows[i].record_index].is_positive]
    negatives = [i for i in kept if not records[windows[i].record_index].is_positive]
    if not positives or not negatives:
        order = list(kept)
        rng.shuffle(order)
        return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]

    per_batch_positive = max(1, int(round(batch_size * positive_fraction)))
    per_batch_negative = max(1, batch_size - per_batch_positive)
    negatives_by_domain = _group_by_domain(negatives, windows, records)

    rng.shuffle(positives)
    batches = []
    # One pass over the positives, which are the scarce class.
    for start in range(0, len(positives), per_batch_positive):
        chosen = positives[start : start + per_batch_positive]
        chosen = chosen + _draw_domain_proportional(negatives_by_domain, per_batch_negative, rng)
        rng.shuffle(chosen)
        batches.append(chosen)
    return batches


def _cap_per_prompt(
    windows: Sequence[Window],
    records: Sequence,
    cap: Optional[int],
    rng: np.random.Generator,
) -> List[int]:
    indices = list(range(len(windows)))
    if cap is None:
        return indices
    rng.shuffle(indices)
    seen: Dict[str, int] = {}
    kept = []
    for index in indices:
        prompt = records[windows[index].record_index].prompt_id
        if seen.get(prompt, 0) >= cap:
            continue
        seen[prompt] = seen.get(prompt, 0) + 1
        kept.append(index)
    return sorted(kept)


def _group_by_domain(
    indices: Sequence[int], windows: Sequence[Window], records: Sequence
) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for index in indices:
        grouped.setdefault(records[windows[index].record_index].domain, []).append(index)
    return grouped


def _draw_domain_proportional(
    grouped: Dict[str, List[int]], count: int, rng: np.random.Generator
) -> List[int]:
    """Draw with each domain represented in proportion to its share."""
    domains = sorted(grouped)
    sizes = np.array([len(grouped[domain]) for domain in domains], dtype=np.float64)
    if sizes.sum() == 0:
        return []
    probabilities = sizes / sizes.sum()
    picks = rng.choice(len(domains), size=count, p=probabilities)
    return [grouped[domains[pick]][rng.integers(0, len(grouped[domains[pick]]))] for pick in picks]
