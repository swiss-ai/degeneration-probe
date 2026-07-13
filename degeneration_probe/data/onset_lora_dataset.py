"""Subtask G: LoRA-condition item construction for the onset probes (`probe_N` /
multi-horizon), restricted to the curated pilot subsample.

Unlike the no-LoRA path (`degeneration_probe.data.onset_dataset`, which reads cached
activations directly), the LoRA condition needs a live forward pass -- cached
activations are invalid once the model's weights change. That means going through
`TokenizedProbingDataset` (`degeneration_probe/data/dataset.py`), which expects a
list of `ProbingItem`s. This module builds those items directly from the dataset-gen
pipeline's parquet files (`generations/<domain>/shard_00000.parquet`,
`prompts/prompts.parquet`, plus subtask A's `onset_labels.parquet`) -- there is no HF
Hub upload of this data, so `degeneration_probe.data.converters.get_prepare_function`'s
generic HF-dataset path doesn't apply here.

**The retokenization gotcha** (see the implementation prompt's "Existing code to
reuse" section) is resolved by setting `ProbingItem.completion_token_ids` to the
rollout's *original* `generated_token_ids` -- `TokenizedProbingDataset
._process_token_level_item` now uses those directly instead of retokenizing
`item.completion`'s decoded text, which isn't guaranteed to round-trip losslessly.

**Label encoding** (`ProbingItem.token_labels`, one float per completion token):
this stores a *continuous distance-to-onset* signal, not a single horizon's binary
label, so the SAME items work for every horizon (and for the eventual multi-horizon
loss in subtask H) without duplicating the dataset per horizon:
    - Positive rollout, eligible position (`t <= onset_position`):
      `token_labels[t] = onset_position - t` (>= 0).
    - Positive rollout, excluded position (`t > onset_position`, i.e. already at/after
      the degenerate span -- out of scope for onset-horizon training, see the
      implementation prompt): `token_labels[t] = IGNORE_LABEL` (-100.0). This reuses
      the *exact* sentinel `TokenizedProbingDataset` already treats as "exclude from
      loss" (`classification_weights[classification_labels == -100.0] = 0.0`), so no
      changes to that shared masking logic were needed.
    - Negative (`stop_reason == "eos"`) rollout, every position:
      `token_labels[t] = NEGATIVE_SENTINEL` (-1.0) -- a value that is never a valid
      non-negative distance, so `0 <= NEGATIVE_SENTINEL <= N` is False for every
      horizon `N >= 0`, i.e. correctly negative at every horizon in one shared value.

Subtask H (multi-horizon LoRA loss) derives any horizon `N`'s binary label directly
from this stored distance via `y_N = (0 <= classification_labels) & (classification_labels <= N)`,
with `classification_weights` (already 0 for `IGNORE_LABEL` positions) reused as-is
for the BCE mask at every horizon.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.types import ProbingItem

IGNORE_LABEL = -100.0
NEGATIVE_SENTINEL = -1.0


def select_lora_pilot_rollouts(
    rollout_index: pd.DataFrame,
    n_negative_target: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Every positive rollout in `rollout_index` plus a domain-stratified sample of
    `n_negative_target` negative rollouts (proportional to each domain's share of the
    negative pool) -- the "matched negative sample stratified by domain" sizing the
    implementation prompt recommends for the LoRA pilot (862 positive rollouts
    corpus-wide + 2,000-3,000 matched negatives is the overall target; call this once
    per split with a `n_negative_target` sized to that split's share of the total).
    """
    positive = rollout_index[rollout_index["is_positive"]]
    negative_pool = rollout_index[~rollout_index["is_positive"]]

    rng = np.random.default_rng(seed)
    sampled_negative_frames = []
    for domain, group in negative_pool.groupby("domain", sort=False):
        domain_share = len(group) / len(negative_pool)
        n_domain = min(len(group), round(n_negative_target * domain_share))
        chosen = rng.choice(len(group), size=n_domain, replace=False)
        sampled_negative_frames.append(group.iloc[chosen])
    sampled_negative = pd.concat(sampled_negative_frames, ignore_index=True) if sampled_negative_frames else negative_pool.iloc[:0]

    return pd.concat([positive, sampled_negative], ignore_index=True)


def _distance_labels(num_tokens: int, is_positive: bool, onset_position: Optional[int]) -> list:
    if not is_positive:
        return [NEGATIVE_SENTINEL] * num_tokens
    onset = int(onset_position)
    return [float(onset - t) if t <= onset else IGNORE_LABEL for t in range(num_tokens)]


def build_onset_probing_items(config: DatasetGenConfig, selected_rollouts: pd.DataFrame) -> list:
    """One `ProbingItem` per row of `selected_rollouts` (as returned by
    `select_lora_pilot_rollouts` or any subset of `onset_dataset.build_rollout_index`'s
    output). Reads `prompts.parquet` once and each involved domain's
    `generations/<domain>/shard_00000.parquet` once (not per rollout), then joins.
    """
    prompts_df = pd.read_parquet(paths.prompts_path(config), columns=["prompt_id", "prompt_text"])
    prompt_id_to_text: Dict[str, str] = dict(zip(prompts_df["prompt_id"], prompts_df["prompt_text"]))

    items = []
    for domain, group in selected_rollouts.groupby("domain", sort=False):
        generations_df = pd.read_parquet(
            paths.generations_shard_path(config, domain, 0),
            columns=["prompt_id", "rollout_idx", "generated_text", "generated_token_ids"],
        )
        merged = group.merge(generations_df, on=["prompt_id", "rollout_idx"], how="inner")
        if len(merged) != len(group):
            raise ValueError(
                f"[{domain}] rollout_index/generations merge mismatch: "
                f"{len(group)} selected rows vs {len(merged)} merged rows"
            )

        for row in merged.itertuples(index=False):
            token_ids = list(row.generated_token_ids)
            token_labels = _distance_labels(
                num_tokens=len(token_ids),
                is_positive=bool(row.is_positive),
                onset_position=row.onset_position if row.is_positive else None,
            )
            items.append(
                ProbingItem(
                    prompt=prompt_id_to_text[row.prompt_id],
                    completion=row.generated_text,
                    spans=[],
                    token_labels=token_labels,
                    completion_token_ids=token_ids,
                )
            )
    return items
