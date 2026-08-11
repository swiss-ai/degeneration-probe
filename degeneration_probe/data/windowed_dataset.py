"""Training examples as windows, drawn by the configured selection rule.

One item is one window: the cached hidden states at the selected token
positions, and the targets for those same positions. Because every rule after
the reference one spends the same budget per rollout, items have a fixed size
and a batch has a fixed cost, so adding a domain or a longer rollout changes
neither.

Windows are redrawn each epoch rather than materialized once. That is free
augmentation, and it makes the comparison between placement rules a comparison
over the pool rather than over one arbitrary draw.

Batch composition is folded into the item order: the dataset hands back items in
the order the composition rule produced, so reading it sequentially in
fixed-size chunks yields exactly the composed batches, each with its class mix
and domain spread intact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from degeneration_probe.data.activation_store import load_probe_layers
from degeneration_probe.data.sampling import build_windows, compose_batches


class WindowedActivationDataset(Dataset):
    """Windows over cached activations, redrawn on demand."""

    def __init__(
        self,
        records: Sequence,
        *,
        build_root: str,
        probe_layer: int,
        selection,
        batch_size: int,
        seed: int = 42,
        compose: bool = True,
        hardness: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        self.records = list(records)
        self.build_root = str(build_root)
        # One depth or many: a sequence trains a head per layer from a single
        # read, since the seek dominates the cost of the file.
        self.probe_layers = (
            [int(probe_layer)]
            if isinstance(probe_layer, (int, np.integer))
            else [int(layer) for layer in probe_layer]
        )
        self.probe_layer = self.probe_layers[0]
        self.selection = selection
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.compose = compose
        self.hardness = hardness
        self.windows: List = []
        self.order: List[int] = []
        self.resample(0)

    def resample(self, epoch: int) -> None:
        """Redraw the windows, and the order batches will read them in."""
        rng = np.random.default_rng([self.seed, epoch])
        self.windows = build_windows(
            self.records,
            strategy=self.selection.strategy,
            window_size=self.selection.window_size,
            rng=rng,
            anchor=self.selection.anchor,
            hard_negative_fraction=self.selection.hard_negative_fraction,
            hardness=self.hardness,
        )
        if not self.compose:
            self.order = list(range(len(self.windows)))
            return
        batches = compose_batches(
            self.windows,
            self.records,
            batch_size=self.batch_size,
            positive_fraction=self.selection.positive_fraction,
            rng=rng,
            max_rollouts_per_prompt=self.selection.max_rollouts_per_prompt,
        )
        self.order = [index for batch in batches for index in batch]
        self._assert_both_classes_present()

    def _assert_both_classes_present(self) -> None:
        """A selection rule and a label family can silently exclude a class.

        A trailing window ends at the frontier, so it holds only run-up tokens;
        paired with a label that marks nothing before the frontier, every
        selected token is negative and the run trains on no positive example at
        all. That converges to a constant and looks like a weak result rather
        than a misconfiguration, so it is refused here instead.
        """
        positive = 0
        for index in self.order:
            window = self.windows[index]
            targets = np.asarray(self.records[window.record_index].targets, dtype=np.float32)[
                window.positions
            ]
            finite = targets[np.isfinite(targets)]
            positive += int((finite > 0).sum())
            if positive:
                return
        raise ValueError(
            f"selection '{self.selection.strategy}' with anchor '{self.selection.anchor}' "
            "selected no positive tokens. A trailing window holds only the run-up, so it "
            "needs a label that marks it: raise training.label.horizon, use "
            "training.label.family=frontier_soft, or anchor the window with "
            "training.selection.anchor=centered."
        )

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> Dict[str, object]:
        window = self.windows[self.order[index]]
        record = self.records[window.record_index]
        positions = window.positions
        features = load_probe_layers(
            self.build_root,
            record.domain,
            record.prompt_id,
            record.rollout_idx,
            probe_layers=self.probe_layers,
        )
        # The window's tokens are taken while the depths are still the leading
        # axis, because that keeps the copy to the window rather than the whole
        # rollout: a permute first would make the same selection strided over
        # every layer of every token.
        features = features[:, positions]
        # [layers, window, hidden] to [window, hidden] for one depth, or
        # [window, layers, hidden] when several share the pass.
        features = features[0] if len(self.probe_layers) == 1 else features.permute(1, 0, 2)
        targets = torch.from_numpy(
            np.asarray(record.targets, dtype=np.float32)[positions]
        )
        return {
            "features": features.to(torch.float32),
            "targets": targets,
            "target_mask": torch.isfinite(targets),
            "prompt_id": record.prompt_id,
            "rollout_idx": record.rollout_idx,
            "domain": record.domain,
            "split": record.split,
            "is_positive": record.is_positive,
            "prompt_length": 0,
        }

    def summary(self) -> Dict[str, object]:
        """Composition of what training will actually see, not of the pool."""
        domains: Dict[str, int] = {}
        positive_rollouts = valid = positive = total = 0
        for index in self.order:
            window = self.windows[index]
            record = self.records[window.record_index]
            domains[record.domain] = domains.get(record.domain, 0) + 1
            targets = np.asarray(record.targets, dtype=np.float32)[window.positions]
            finite = np.isfinite(targets)
            valid += int(finite.sum())
            total += int(targets.size)
            hits = int((targets[finite] > 0).sum())
            positive += hits
            positive_rollouts += int(hits > 0)
        return {
            "windows": len(self.order),
            "rollouts": len({self.windows[i].record_index for i in self.order}),
            "positive_rollouts": positive_rollouts,
            "negative_rollouts": len(self.order) - positive_rollouts,
            "tokens": total,
            "valid_tokens": valid,
            "positive_tokens": positive,
            "negative_tokens": valid - positive,
            "positive_token_rate": positive / valid if valid else 0.0,
            "rollouts_per_domain": dict(sorted(domains.items())),
        }

    def target_counts(self):
        negative = positive = 0
        for index in self.order:
            window = self.windows[index]
            targets = np.asarray(self.records[window.record_index].targets, dtype=np.float32)[
                window.positions
            ]
            finite = targets[np.isfinite(targets)]
            positive += int((finite == 1.0).sum())
            negative += int((finite == 0.0).sum())
            if not np.all((finite == 0.0) | (finite == 1.0)):
                raise ValueError("A class weight needs targets that are exactly 0 or 1")
        return negative, positive
