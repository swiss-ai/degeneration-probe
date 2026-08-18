"""Training examples read from the activation cache instead of a live model.

With the language model frozen, the hidden state at every token of every
rollout is a constant that was computed once already. Training then needs no
forward pass at all: it reads vectors and fits a linear head, which is bounded
by input and output rather than by compute. That is what makes a sweep over
recipes, window sizes and seeds affordable enough to run exhaustively.

This regime is only valid while nothing adapts the model. The moment adapters
are trained, the cached vectors describe a model that no longer exists, so the
configuration refuses the combination rather than quietly training on stale
features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from degeneration_probe.config import DatasetConfig, TokenizationConfig
from degeneration_probe.data.activation_store import load_probe_layers
from degeneration_probe.data.dataset import (
    DegenerationRecord,
    load_degeneration_records,
)


class CachedActivationDataset(Dataset):
    """One rollout per item: its cached layer, and its per-token targets."""

    def __init__(
        self,
        records: List[DegenerationRecord],
        *,
        activations_root: str,
        probe_layer: int,
        tokenization: TokenizationConfig,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.activations_root = str(activations_root)
        # One depth or many: a sequence trains a head per layer from a single
        # read, since the seek dominates the cost of the file.
        self.probe_layers = (
            [int(probe_layer)]
            if isinstance(probe_layer, (int, np.integer))
            else [int(layer) for layer in probe_layer]
        )
        self.probe_layer = self.probe_layers[0]
        self.tokenization = tokenization
        if shuffle:
            order = np.random.default_rng(seed).permutation(len(self.records))
            self.records = [self.records[index] for index in order]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        targets = np.asarray(
            record.targets[: self.tokenization.max_completion_length], dtype=np.float32
        )
        features = load_probe_layers(
            self.activations_root,
            record.domain,
            record.prompt_id,
            record.rollout_idx,
            probe_layers=self.probe_layers,
        )
        # [layers, tokens, hidden] to [tokens, hidden] for one depth, or
        # [tokens, layers, hidden] when several share the pass.
        features = features[0] if len(self.probe_layers) == 1 else features.permute(1, 0, 2)[: len(targets)]
        targets = torch.from_numpy(targets[: features.shape[0]])
        return {
            "features": features.to(torch.float32),
            "targets": targets,
            "target_mask": torch.isfinite(targets),
            "prompt_id": record.prompt_id,
            "rollout_idx": record.rollout_idx,
            "domain": record.domain,
            "split": record.split,
            "is_positive": record.is_positive,
            # Where the loop starts, carried through so that evaluation can
            # measure coverage of the run-up without re-deriving a frontier from
            # the targets, which a soft label makes ambiguous.
            "onset_position": record.onset_position,
            # No prompt is prepended here: the cache holds completion tokens only.
            "prompt_length": 0,
        }

    def summary(self) -> Dict[str, object]:
        """Composition of the split, in the same shape the live dataset reports."""
        limit = self.tokenization.max_completion_length
        domains: Dict[str, int] = {}
        positive_rollouts = valid_tokens = positive_tokens = total_tokens = 0
        for record in self.records:
            domains[record.domain] = domains.get(record.domain, 0) + 1
            targets = np.asarray(record.targets[:limit], dtype=np.float32)
            finite = np.isfinite(targets)
            valid_tokens += int(finite.sum())
            total_tokens += int(len(targets))
            positives = int((targets[finite] > 0).sum())
            positive_tokens += positives
            positive_rollouts += int(positives > 0)
        return {
            "rollouts": len(self.records),
            "positive_rollouts": positive_rollouts,
            "negative_rollouts": len(self.records) - positive_rollouts,
            "tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "positive_tokens": positive_tokens,
            "negative_tokens": valid_tokens - positive_tokens,
            "positive_token_rate": positive_tokens / valid_tokens if valid_tokens else 0.0,
            "rollouts_per_domain": dict(sorted(domains.items())),
        }

    def target_counts(self):
        negative = positive = 0
        limit = self.tokenization.max_completion_length
        for record in self.records:
            targets = np.asarray(record.targets[:limit], dtype=np.float32)
            finite = targets[np.isfinite(targets)]
            positive += int((finite == 1.0).sum())
            negative += int((finite == 0.0).sum())
            if not np.all((finite == 0.0) | (finite == 1.0)):
                raise ValueError("A class weight needs targets that are exactly 0 or 1")
        return negative, positive


def cached_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Pad to the longest rollout in the batch, masking what was added."""
    max_length = max(item["features"].shape[0] for item in batch)
    # Everything after the token axis is carried through unchanged, which is
    # what lets a probe reading several depths at once use the same collate: its
    # features simply carry a layer axis before the hidden one.
    trailing = tuple(batch[0]["features"].shape[1:])
    features = torch.zeros((len(batch), max_length, *trailing), dtype=torch.float32)
    targets = torch.full((len(batch), max_length), float("nan"), dtype=torch.float32)
    target_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["features"].shape[0]
        features[row, :length] = item["features"]
        targets[row, :length] = item["targets"]
        target_mask[row, :length] = item["target_mask"]
    return {
        "features": features,
        "targets": targets,
        "target_mask": target_mask,
        "prompt_length": [0] * len(batch),
        "prompt_id": [item["prompt_id"] for item in batch],
        "rollout_idx": [item["rollout_idx"] for item in batch],
        "domain": [item["domain"] for item in batch],
        "split": [item["split"] for item in batch],
        "is_positive": [item.get("is_positive", False) for item in batch],
        "onset_position": [item.get("onset_position") for item in batch],
    }


def create_cached_dataset(
    config: DatasetConfig,
    *,
    split: str,
    label_config,
    probe_layer: int,
    training: bool,
) -> CachedActivationDataset:
    records = load_degeneration_records(
        config, split=split, label_config=label_config, training=training
    )
    return CachedActivationDataset(
        records,
        activations_root=str(config.activations_dir),
        probe_layer=probe_layer,
        tokenization=config.tokenization,
        shuffle=training,
        seed=config.sampling.seed,
    )
