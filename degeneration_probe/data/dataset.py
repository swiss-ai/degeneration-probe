"""Local Parquet loader and token-aligned dataset for degeneration training."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from degeneration_probe.config import DatasetConfig, TokenizationConfig
from degeneration_probe.data.labels import derive_targets, required_signal

REPO_ROOT = Path(__file__).resolve().parents[2]

# Evaluation populations are drawn from the split alone, never from the run
# seed: a seed repeat must be judged on identical rows, or its spread measures
# the sample rather than the run.
EVALUATION_SEED = 0


@dataclass(frozen=True)
class DegenerationRecord:
    prompt_id: str
    rollout_idx: int
    domain: str
    split: str
    prompt_text: str
    generated_token_ids: Sequence[int]
    targets: Sequence[float]
    # Carried so a selection rule can anchor on the frontier without
    # re-deriving it from the targets, which a soft label makes ambiguous.
    is_positive: bool = False
    onset_position: Optional[int] = None


def derive_bce_targets(
    num_tokens: int,
    *,
    stop_reason: str,
    onset_position: Optional[float],
) -> Optional[List[float]]:
    """Build the binary target, or return ``None`` for an unusable rollout."""
    if stop_reason == "eos":
        return [0.0] * num_tokens
    if onset_position is None or pd.isna(onset_position):
        return None
    onset = int(onset_position)
    if onset < 0 or onset >= num_tokens:
        raise ValueError(f"onset_position={onset} is outside a {num_tokens}-token rollout")
    return [0.0] * onset + [1.0] * (num_tokens - onset)


def derive_mse_targets(repetition_score: Sequence[float], num_tokens: int) -> List[float]:
    """Align the original scores to token IDs; unmatched positions stay masked."""
    targets = [float("nan")] * num_tokens
    for index, value in enumerate(list(repetition_score)[:num_tokens]):
        if value is not None:
            targets[index] = float(value)
    return targets


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)


def select_rollouts(
    onset_labels: pd.DataFrame,
    *,
    split: str,
    negative_rollouts_per_positive: Optional[float],
    domain_stratified: bool,
    seed: int,
) -> pd.DataFrame:
    """Keep defined-onset rollouts and a deterministic sample of EOS rollouts."""
    split_rows = onset_labels[onset_labels["split"] == split].copy()
    positive_mask = split_rows["is_positive"].astype(bool) & split_rows["onset_position"].notna()
    negative_mask = split_rows["stop_reason"].eq("eos")
    positives = split_rows[positive_mask]
    negatives = split_rows[negative_mask]

    if negative_rollouts_per_positive is None:
        sampled_negatives = negatives
    else:
        target = min(len(negatives), round(len(positives) * negative_rollouts_per_positive))
        if target == 0:
            sampled_negatives = negatives.iloc[:0]
        elif not domain_stratified:
            sampled_negatives = negatives.sample(n=target, random_state=_stable_seed(seed, split))
        else:
            # Allocate proportionally by domain using the largest-remainder method.
            sizes = negatives.groupby("domain", sort=True).size()
            quotas = sizes / sizes.sum() * target
            allocations = np.floor(quotas).astype(int)
            remainder = target - int(allocations.sum())
            for domain in (quotas - allocations).sort_values(ascending=False).index[:remainder]:
                allocations.loc[domain] += 1
            frames = []
            for domain, count in allocations.items():
                if count:
                    group = negatives[negatives["domain"] == domain]
                    frames.append(
                        group.sample(
                            n=min(int(count), len(group)),
                            random_state=_stable_seed(seed, split, str(domain)),
                        )
                    )
            sampled_negatives = pd.concat(frames, ignore_index=False) if frames else negatives.iloc[:0]

    selected = pd.concat([positives, sampled_negatives], ignore_index=True)
    if selected.empty:
        return selected
    return selected.sort_values(["domain", "prompt_id", "rollout_idx"], kind="stable").reset_index(drop=True)


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _source_names(config: dict) -> set[str]:
    return {
        source["name"]
        for key in ("in_domain_sources", "held_out_sources")
        for source in config.get(key, [])
    }


def validate_build_artifacts(config: DatasetConfig) -> None:
    """Fail on missing artifacts and warn when generation metadata is stale."""
    root = Path(config.build_root)
    prompts = root / "prompts" / "prompts.parquet"
    onset = root / "onset_labels" / "onset_labels.parquet"
    for required in (prompts, onset):
        if not required.is_file():
            raise FileNotFoundError(f"Required dataset artifact not found: {required}")

    onset_df = pd.read_parquet(onset, columns=["domain"])
    for domain in sorted(onset_df["domain"].dropna().unique()):
        for artifact in ("generations", "labels"):
            shards = sorted((root / artifact / str(domain)).glob("*.parquet"))
            if not shards:
                raise FileNotFoundError(f"No {artifact} Parquet shard found for domain {domain!r}")

    build_config_path = _resolve_repo_path(config.build_config)
    manifest_path = root / "manifest.json"
    if not build_config_path.is_file() or not manifest_path.is_file():
        missing = build_config_path if not build_config_path.is_file() else manifest_path
        warnings.warn(f"Cannot compare dataset metadata because {missing} is missing", stacklevel=2)
        return

    import yaml

    with build_config_path.open("r", encoding="utf-8") as handle:
        expected = yaml.safe_load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        materialized = json.load(handle).get("config", {})
    mismatch = (
        expected.get("model_name") != materialized.get("model_name")
        or _source_names(expected) != _source_names(materialized)
    )
    if mismatch:
        warnings.warn(
            "The build YAML and manifest.json disagree. Required artifacts were verified; "
            "onset_labels.parquet remains the source of truth for domains and splits.",
            stacklevel=2,
        )


def _read_selected_domain_shards(
    root: Path,
    artifact: str,
    selected: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Read one domain at a time and immediately discard unselected rows."""
    frames = []
    keys = ["prompt_id", "rollout_idx"]
    for domain in sorted(selected["domain"].astype(str).unique()):
        selected_keys = selected.loc[selected["domain"].astype(str) == domain, keys]
        shard_paths = sorted((root / artifact / domain).glob("*.parquet"))
        for path in shard_paths:
            shard = pd.read_parquet(path, columns=columns)
            frames.append(shard.merge(selected_keys, on=keys, how="inner"))
    return pd.concat(frames, ignore_index=True)


def _assert_unique(frame: pd.DataFrame, columns: List[str], name: str) -> None:
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, columns].head().to_dict("records")
        raise ValueError(f"Duplicate {name} keys for {columns}: {sample}")


def load_degeneration_records(
    config: DatasetConfig,
    *,
    split: str,
    label_config,
    training: bool,
) -> List[DegenerationRecord]:
    """Join all local artifacts on ``(prompt_id, rollout_idx)``."""
    validate_build_artifacts(config)
    root = Path(config.build_root)
    onset = pd.read_parquet(root / "onset_labels" / "onset_labels.parquet")
    ratio = (
        config.sampling.train_negative_rollouts_per_positive
        if training
        else config.sampling.evaluation_negative_rollouts_per_positive
    )
    selected = select_rollouts(
        onset,
        split=split,
        negative_rollouts_per_positive=ratio,
        domain_stratified=config.sampling.domain_stratified,
        seed=config.sampling.seed if training else EVALUATION_SEED,
    )
    if selected.empty:
        raise ValueError(f"No usable rollouts found for split {split!r}")

    keys = ["prompt_id", "rollout_idx"]
    _assert_unique(selected, keys, "onset label")
    prompts = pd.read_parquet(
        root / "prompts" / "prompts.parquet", columns=["prompt_id", "prompt_text"]
    )
    selected_prompt_ids = set(selected["prompt_id"])
    prompts = prompts[prompts["prompt_id"].isin(selected_prompt_ids)].reset_index(drop=True)
    generations = _read_selected_domain_shards(
        root,
        "generations",
        selected,
        ["prompt_id", "rollout_idx", "generated_token_ids"],
    )
    # Only a family that reads a per-token signal needs the label shards at all.
    signal_column = required_signal(label_config)
    labels = (
        _read_selected_domain_shards(
            root, "labels", selected, ["prompt_id", "rollout_idx", signal_column]
        )
        if signal_column
        else None
    )
    _assert_unique(prompts, ["prompt_id"], "prompt")
    _assert_unique(generations, keys, "generation")
    if labels is not None:
        _assert_unique(labels, keys, "label")

    merged = selected.merge(prompts, on="prompt_id", how="left", validate="many_to_one")
    merged = merged.merge(generations, on=keys, how="left", validate="one_to_one")
    if labels is not None:
        merged = merged.merge(labels, on=keys, how="left", validate="one_to_one")
    required_columns = ["prompt_text", "generated_token_ids"] + (
        [signal_column] if signal_column else []
    )
    if merged[required_columns].isna().any().any():
        bad = merged.loc[merged[required_columns].isna().any(axis=1), keys].head().to_dict("records")
        raise ValueError(f"Dataset join left missing artifacts for rollout keys: {bad}")

    records: List[DegenerationRecord] = []
    for row in merged.itertuples(index=False):
        token_ids = np.asarray(row.generated_token_ids, dtype=np.int64)
        targets = derive_targets(
            label_config,
            num_tokens=len(token_ids),
            stop_reason=row.stop_reason,
            onset_position=row.onset_position,
            signal=getattr(row, signal_column) if signal_column else None,
        )
        if targets is None:  # A rollout the family cannot label is dropped, not invented.
            continue
        targets = np.asarray(targets, dtype=np.float32)
        if not np.isfinite(targets).any():
            continue
        records.append(
            DegenerationRecord(
                prompt_id=str(row.prompt_id),
                rollout_idx=int(row.rollout_idx),
                domain=str(row.domain),
                split=str(row.split),
                prompt_text=str(row.prompt_text),
                generated_token_ids=token_ids,
                targets=targets,
                is_positive=bool(row.stop_reason != "eos" and pd.notna(row.onset_position)),
                onset_position=int(row.onset_position)
                if pd.notna(row.onset_position)
                else None,
            )
        )
    return records


def load_token_signal(
    config: DatasetConfig,
    records: Sequence["DegenerationRecord"],
    column: str = "repetition_score",
) -> Dict[int, np.ndarray]:
    """A per-token signal for the given rollouts, keyed by their index.

    Hard-negative placement needs to know where a rollout looks repetitive,
    which is a different question from what it is labelled with, so the signal
    is loaded on its own rather than through the label family.
    """
    if not records:
        return {}
    root = Path(config.build_root)
    keys = pd.DataFrame(
        {
            "prompt_id": [record.prompt_id for record in records],
            "rollout_idx": [record.rollout_idx for record in records],
            "domain": [record.domain for record in records],
        }
    )
    frame = _read_selected_domain_shards(
        root, "labels", keys, ["prompt_id", "rollout_idx", column]
    )
    lookup = {
        (row.prompt_id, int(row.rollout_idx)): np.asarray(
            getattr(row, column), dtype=np.float64
        )
        for row in frame.itertuples(index=False)
    }
    signal: Dict[int, np.ndarray] = {}
    for index, record in enumerate(records):
        values = lookup.get((record.prompt_id, record.rollout_idx))
        if values is None:
            continue
        aligned = np.zeros(len(record.targets), dtype=np.float64)
        usable = min(len(values), aligned.size)
        aligned[:usable] = np.nan_to_num(values[:usable], nan=0.0)
        signal[index] = aligned
    return signal


class DegenerationTokenDataset(Dataset):
    """Prepend a tokenized prompt while preserving generated token IDs exactly.

    A selection rule can be applied here too, but it means something different
    from the cached regime. Attention is causal, so producing the hidden state
    at a token requires running the whole prefix before it: a window cannot
    make the forward pass cheaper, only decide which positions contribute to
    the loss. The rung is therefore expressible while the model adapts, but as
    a mask rather than as a smaller batch, and per-batch class composition does
    not carry over, since the unit here is a rollout rather than a window.
    """

    def __init__(
        self,
        records: List[DegenerationRecord],
        tokenizer,
        tokenization: TokenizationConfig,
        *,
        shuffle: bool = False,
        seed: int = 42,
        selection=None,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.tokenization = tokenization
        self.selection = selection
        # Only actually consumed below (shuffle) and in resample() (selection
        # not None) -- eval-time callers (shuffle=False, selection=None) never
        # touch it, so a dataset config with sampling.seed unset must not
        # crash a scoring run that never shuffles or resamples.
        self.seed = seed
        self._selected: Optional[Dict[int, np.ndarray]] = None
        if shuffle:
            order = np.random.default_rng(int(seed)).permutation(len(self.records))
            self.records = [self.records[index] for index in order]
        if self.selection is not None:
            self.resample(0)

    def resample(self, epoch: int) -> None:
        """Redraw which positions contribute to the loss."""
        from degeneration_probe.data.sampling import build_windows

        rng = np.random.default_rng([int(self.seed), epoch])
        windows = build_windows(
            self.records,
            strategy=self.selection.strategy,
            window_size=self.selection.window_size,
            rng=rng,
            anchor=self.selection.anchor,
            hard_negative_fraction=self.selection.hard_negative_fraction,
        )
        selected: Dict[int, List[np.ndarray]] = {}
        for window in windows:
            selected.setdefault(window.record_index, []).append(window.positions)
        self._selected = {
            index: np.unique(np.concatenate(parts)) for index, parts in selected.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def _prompt_ids(self, prompt: str) -> List[int]:
        conversation = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        if self.tokenizer.bos_token and self.tokenizer.bos_token in text:
            text = text.replace(self.tokenizer.bos_token, "")
        return list(
            self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        completion_ids = record.generated_token_ids[: self.tokenization.max_completion_length]
        completion_targets = record.targets[: len(completion_ids)]
        prompt_ids = self._prompt_ids(record.prompt_text)
        prompt_budget = self.tokenization.max_length - len(completion_ids)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = (
                prompt_ids[-prompt_budget:]
                if self.tokenization.prompt_truncation_side == "left" and prompt_budget
                else prompt_ids[:prompt_budget]
            )

        input_ids = torch.tensor(prompt_ids + list(completion_ids), dtype=torch.long)
        targets = torch.full((len(input_ids),), float("nan"), dtype=torch.float32)
        if len(completion_targets):
            targets[len(prompt_ids) :] = torch.tensor(completion_targets, dtype=torch.float32)
        target_mask = torch.isfinite(targets)
        if self._selected is not None:
            keep = torch.zeros_like(target_mask)
            positions = self._selected.get(index, np.empty(0, dtype=np.int64))
            positions = positions[positions < len(completion_targets)]
            keep[len(prompt_ids) + torch.as_tensor(positions, dtype=torch.long)] = True
            target_mask &= keep
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "targets": targets,
            "target_mask": target_mask,
            # Where the completion starts, so a scorer can slice out exactly the
            # generated tokens without inferring it from which targets are defined.
            "prompt_length": len(prompt_ids),
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0),
            "prompt_id": record.prompt_id,
            "rollout_idx": record.rollout_idx,
            "domain": record.domain,
            "split": record.split,
            "is_positive": record.is_positive,
        }

    def _effective_targets(self, index: int) -> np.ndarray:
        """Targets as the loss will see them, with any selection rule applied.

        A rule applied while the model adapts masks the loss rather than
        shrinking the batch, so the population a run actually trains on is the
        masked one, not the pool it was drawn from. Reading composition off the
        pool would report a positive rate and a class weight belonging to a
        recipe that was never run.
        """
        record = self.records[index]
        limit = min(
            self.tokenization.max_completion_length, len(record.generated_token_ids)
        )
        targets = np.asarray(record.targets[:limit], dtype=np.float32)
        if self._selected is None:
            return targets
        keep = np.zeros(targets.shape, dtype=bool)
        positions = self._selected.get(index, np.empty(0, dtype=np.int64))
        keep[positions[positions < targets.size]] = True
        return np.where(keep, targets, np.float32("nan"))

    def summary(self) -> Dict[str, object]:
        """Composition of the split as the probe will actually see it.

        Recorded per run so that a later comparison can tell an effect of the
        recipe from an effect of how much data the recipe was given, and so
        that the realized positive rate stays visible next to any class weight
        applied on top of it.
        """
        domains: Dict[str, int] = {}
        positive_rollouts = 0
        valid_tokens = 0
        positive_tokens = 0
        total_tokens = 0
        for index, record in enumerate(self.records):
            domains[record.domain] = domains.get(record.domain, 0) + 1
            targets = self._effective_targets(index)
            finite = np.isfinite(targets)
            valid_tokens += int(finite.sum())
            total_tokens += int(targets.size)
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

    def target_counts(self) -> Tuple[int, int]:
        """Return negative and positive valid-token counts for BCE weighting.

        Counted over the masked population, so the weight corrects the skew the
        run is actually exposed to.
        """
        negative = positive = 0
        for index in range(len(self.records)):
            targets = self._effective_targets(index)
            finite = targets[np.isfinite(targets)]
            positives = int((finite == 1.0).sum())
            negatives = int((finite == 0.0).sum())
            if positives + negatives != finite.size:
                raise ValueError("BCE targets must be exactly 0 or 1")
            positive += positives
            negative += negatives
        return negative, positive


def degeneration_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_length = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)
    pad_token_id = int(batch[0]["pad_token_id"])
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    targets = torch.full((batch_size, max_length), float("nan"), dtype=torch.float32)
    target_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        targets[row, :length] = item["targets"]
        target_mask[row, :length] = item["target_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "target_mask": target_mask,
        "prompt_length": [item["prompt_length"] for item in batch],
        "prompt_id": [item["prompt_id"] for item in batch],
        "rollout_idx": [item["rollout_idx"] for item in batch],
        "domain": [item["domain"] for item in batch],
        "split": [item["split"] for item in batch],
        "is_positive": [item.get("is_positive", False) for item in batch],
    }


def create_degeneration_dataset(
    config: DatasetConfig,
    tokenizer,
    *,
    split: str,
    label_config,
    training: bool,
    selection=None,
) -> DegenerationTokenDataset:
    records = load_degeneration_records(
        config, split=split, label_config=label_config, training=training
    )
    return DegenerationTokenDataset(
        records,
        tokenizer,
        config.tokenization,
        shuffle=training,
        seed=config.sampling.seed,
        selection=selection,
    )


def compute_pos_weight(dataset: DegenerationTokenDataset) -> float:
    negative, positive = dataset.target_counts()
    if positive == 0:
        raise ValueError("Cannot compute BCE pos_weight: the training set has no positive tokens")
    return negative / positive
