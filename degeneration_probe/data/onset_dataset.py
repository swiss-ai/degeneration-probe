"""Cached-activation dataset for the no-LoRA onset probes (`probe_N` / multi-horizon).

Reads token-level hidden states directly from the already-cached
`activations/<domain>/<prompt_id>/rollout_<k>.safetensors` files (see
`degeneration_probe.dataset_gen.paths.rollout_activation_path`) at a single chosen
layer, paired with per-horizon onset labels resolved by subtask A
(`degeneration_probe.dataset_gen.onset_labels`). This is deliberately NOT built on
`TokenizedProbingDataset` / `ValueHeadProbe` / `ProbeTrainer` -- those exist for the
live-forward-pass LoRA condition, where cached activations are invalid inputs once
the model's weights change. For the frozen-model (no-LoRA) condition, forcing data
through a live-model path would throw away the entire point of having 100%-cached
activations on disk.

Eligible token positions per rollout (per the implementation prompt's settled
definition):
    - Negative (`stop_reason == "eos"`) rollout: every position `t`, label 0 at
      every horizon.
    - Positive (`stop_reason == "length"` with a defined onset) rollout: positions
      `t in [0, onset_position]` (onset inclusive); `y_N(t) = 1 if (onset - t) <= N
      else 0`.
    - Truncated rollout with NO defined onset under the active metric: excluded
      entirely -- `build_rollout_index` drops these.

Imbalance handling (the corpus is a rare-event problem at every horizon -- see the
implementation prompt's sparsity table, positive rate <= 1.1% even at N=200) has two
independent knobs, deliberately kept separate since they operate at different
granularities:
    - `OnsetActivationDataset(max_negative_tokens_per_rollout=...)`: within one
      negative rollout, don't keep every one of its (possibly thousands of)
      same-label positions -- subsample a fixed number of them.
    - `ImbalancedRolloutSampler(negative_rollouts_per_positive=...)`: across
      rollouts, don't let the ~45x negative:positive rollout ratio dominate epoch
      composition -- every positive rollout is used every epoch, negative rollouts
      are subsampled to a configurable multiple of the positive count.
`pos_weight_for_horizons` additionally exposes the *un-subsampled*, corpus-true
inverse positive rate per horizon, for callers who want to compensate the loss
directly (e.g. `BCEWithLogitsLoss(pos_weight=...)`) rather than (or in addition to)
subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset, Sampler

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.onset_labels import DEFAULT_HORIZONS, eligible_token_counts_per_horizon

HIDDEN_SIZE = 4096
NUM_HIDDEN_STATE_LAYERS = 33  # embedding output (0) + 32 decoder blocks (1..32)


def build_rollout_index(onset_labels_df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rollouts usable for onset-probe training/eval in `split`: every EOS
    (negative) rollout plus every positive rollout with a defined onset. Truncated
    rollouts with no defined onset are excluded entirely -- see `onset_labels.py`'s
    module docstring for why inventing a value for those would be wrong."""
    split_df = onset_labels_df[onset_labels_df["split"] == split]
    usable = split_df[(split_df["stop_reason"] == "eos") | split_df["is_positive"]]
    return usable.reset_index(drop=True)


@dataclass
class RolloutTokens:
    positions: np.ndarray  # [n] token positions actually read/kept from this rollout
    hidden: torch.Tensor  # [n, HIDDEN_SIZE], float32
    labels: torch.Tensor  # [n, len(horizons)], float32 0/1


class OnsetActivationDataset(Dataset):
    """One item == one rollout's selected eligible-token hidden states at a single
    layer, plus per-horizon onset labels for those same positions.

    Rollout granularity, not token granularity: reads the whole eligible contiguous
    token range for the rollout's layer in exactly one `safe_open(...).get_slice(...)`
    call (true partial/mmap'd read, ~18ms even for a several-hundred-token slice --
    see the implementation prompt's verified gotcha about never using `load_file()`
    for this). Reading token-by-token would mean reopening the same file up to
    thousands of times for a single long negative rollout.

    `max_negative_tokens_per_rollout`, if set, randomly subsamples (without
    replacement, positions kept sorted) how many of a *negative* rollout's read
    positions are actually returned -- since every position in a negative rollout
    has the identical all-0 label, subsampling loses no label diversity, only
    redundant near-duplicate hidden states. Positive rollouts always return every
    eligible position; they're the rare, valuable examples.

    Collate with `onset_collate_fn` for a flat `(hidden [total, HIDDEN_SIZE], labels
    [total, K])` batch suitable for a plain linear/MLP probe head -- there is no
    padding/attention-mask machinery here, unlike `TokenizedProbingDataset`, because
    every returned position is already known-valid.
    """

    def __init__(
        self,
        config: DatasetGenConfig,
        rollout_index: pd.DataFrame,
        layer: int,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        max_negative_tokens_per_rollout: Optional[int] = 32,
        seed: int = 0,
    ):
        if not (0 <= layer < NUM_HIDDEN_STATE_LAYERS):
            raise ValueError(
                f"layer must be in [0, {NUM_HIDDEN_STATE_LAYERS - 1}] "
                f"({NUM_HIDDEN_STATE_LAYERS} hidden_states entries: embedding + 32 decoder blocks), "
                f"got {layer}"
            )
        self.config = config
        self.rollout_index = rollout_index.reset_index(drop=True)
        self.layer = layer
        self.horizons = list(horizons)
        self.max_negative_tokens_per_rollout = max_negative_tokens_per_rollout
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rollout_index)

    def __getitem__(self, idx: int) -> RolloutTokens:
        row = self.rollout_index.iloc[idx]
        activation_path = paths.rollout_activation_path(
            self.config, row["domain"], row["prompt_id"], int(row["rollout_idx"])
        )
        is_positive = bool(row["is_positive"])
        eligible_end = int(row["onset_position"]) + 1 if is_positive else int(row["num_tokens"])

        with safe_open(str(activation_path), framework="pt") as f:
            hidden = f.get_slice("hidden_states")[self.layer, 0:eligible_end, :]  # [eligible_end, HIDDEN_SIZE]

        positions = np.arange(eligible_end)
        if (
            not is_positive
            and self.max_negative_tokens_per_rollout is not None
            and eligible_end > self.max_negative_tokens_per_rollout
        ):
            chosen = self._rng.choice(eligible_end, size=self.max_negative_tokens_per_rollout, replace=False)
            chosen.sort()
            positions = chosen
            hidden = hidden[chosen]

        labels = self._labels_for(positions, is_positive, row)
        return RolloutTokens(positions=positions, hidden=hidden.float(), labels=labels)

    def _labels_for(self, positions: np.ndarray, is_positive: bool, row: pd.Series) -> torch.Tensor:
        if not is_positive:
            return torch.zeros((len(positions), len(self.horizons)), dtype=torch.float32)
        onset = int(row["onset_position"])
        distance = onset - positions  # >= 0 for every eligible position, by construction
        labels = np.stack([(distance <= horizon).astype(np.float32) for horizon in self.horizons], axis=1)
        return torch.from_numpy(labels)


def onset_collate_fn(batch: List[RolloutTokens]) -> Dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.cat([item.hidden for item in batch], dim=0),
        "labels": torch.cat([item.labels for item in batch], dim=0),
    }


class ImbalancedRolloutSampler(Sampler[int]):
    """Rollout-level imbalance handling: positive rollouts are rare (e.g. 605/28040
    in `train`, see the implementation prompt's sparsity table), so plain random
    shuffling would make most batches all-negative. Each epoch (each fresh
    `__iter__` call), yields every positive rollout's index plus a fresh random
    sample of `negative_rollouts_per_positive * n_positive` negative rollout
    indices (without replacement within the epoch), then shuffles the combined
    order.
    """

    def __init__(
        self,
        rollout_index: pd.DataFrame,
        negative_rollouts_per_positive: float = 4,
        seed: int = 0,
    ):
        self.positive_indices = rollout_index.index[rollout_index["is_positive"]].to_numpy()
        self.negative_indices = rollout_index.index[~rollout_index["is_positive"]].to_numpy()
        self.negative_rollouts_per_positive = negative_rollouts_per_positive
        self._rng = np.random.default_rng(seed)

    def _n_negative_wanted(self) -> int:
        return min(
            len(self.negative_indices),
            round(self.negative_rollouts_per_positive * max(len(self.positive_indices), 1)),
        )

    def __iter__(self):
        sampled_negative = self._rng.choice(self.negative_indices, size=self._n_negative_wanted(), replace=False)
        epoch_indices = np.concatenate([self.positive_indices, sampled_negative])
        self._rng.shuffle(epoch_indices)
        return iter(epoch_indices.tolist())

    def __len__(self) -> int:
        return len(self.positive_indices) + self._n_negative_wanted()


def materialize_features(
    config: DatasetGenConfig,
    rollout_index: pd.DataFrame,
    layer: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    max_negative_tokens_per_rollout: Optional[int] = 32,
    negative_rollouts_per_positive: Optional[float] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot materialization: read every selected rollout's activations at
    `layer` exactly once into two in-memory tensors, `(X [N, HIDDEN_SIZE], Y [N,
    len(horizons)])`. Every training epoch after this is a pure in-memory tensor
    op -- re-running `OnsetActivationDataset.__getitem__` once per epoch (as a
    naive `DataLoader` loop would) means re-opening thousands of safetensors files
    per epoch for no benefit, since (aside from `max_negative_tokens_per_rollout`'s
    subsample, fixed here by `seed`) the underlying activations never change
    between epochs.

    If `negative_rollouts_per_positive` is given, negative rollouts are first
    subsampled to `negative_rollouts_per_positive * n_positive` via the same
    logic `ImbalancedRolloutSampler` uses (for a *training* set) -- pass `None`
    (the default) to keep every rollout in `rollout_index` (appropriate for a
    fixed *eval* set, where every negative rollout still gets its per-rollout
    token subsample via `max_negative_tokens_per_rollout`, but the rollout
    population itself isn't additionally subsampled).
    """
    if negative_rollouts_per_positive is not None:
        sampler = ImbalancedRolloutSampler(
            rollout_index, negative_rollouts_per_positive=negative_rollouts_per_positive, seed=seed
        )
        selected = rollout_index.loc[list(iter(sampler))].reset_index(drop=True)
    else:
        selected = rollout_index

    dataset = OnsetActivationDataset(
        config, selected, layer=layer, horizons=horizons,
        max_negative_tokens_per_rollout=max_negative_tokens_per_rollout, seed=seed,
    )
    items = [dataset[i] for i in range(len(dataset))]
    X = torch.cat([item.hidden for item in items], dim=0)
    Y = torch.cat([item.labels for item in items], dim=0)
    return X, Y


def pos_weight_for_horizons(
    rollout_index: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> Dict[int, float]:
    """Corpus-true (un-subsampled) inverse positive rate per horizon, for
    `BCEWithLogitsLoss(pos_weight=...)` -- `(n_eligible - n_positive) / n_positive`.
    Computed over whichever rollouts are in `rollout_index` (e.g. one split), via
    the same counting logic subtask A's sanity check uses
    (`eligible_token_counts_per_horizon`), so this always stays consistent with
    those numbers.
    """
    counts_df = eligible_token_counts_per_horizon(rollout_index, horizons=horizons)
    return {
        int(row.horizon): (row.eligible_tokens - row.positive_tokens) / row.positive_tokens
        for row in counts_df.itertuples(index=False)
    }
