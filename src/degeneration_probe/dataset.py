"""PyTorch dataset and collation for degeneration probe training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from degeneration_probe.types import DegenerationItem


class DegenerationDataset(Dataset):
    """
    Dataset that reads labelled generations from a JSONL file.

    Expected JSONL fields (Phase 1 output format):
        prompt, generated_text, degenerating

    Also accepts a simplified format:
        prompt, completion, label
    """

    def __init__(self, paths: str | Path | list[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.items: List[DegenerationItem] = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    record = json.loads(line)
                    prompt = record["prompt"]
                    completion = record.get("generated_text", record.get("completion", ""))
                    label = float(record.get("degenerating", record.get("label", 0)))
                    self.items.append(DegenerationItem(prompt=prompt, completion=completion, label=label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DegenerationItem:
        return self.items[idx]


def make_collate_fn(tokenizer, max_length: int = 2048):
    """
    Return a collate function that tokenizes prompt+completion and builds
    a completion_mask so the probe pools only over completion tokens.

    Returns batches with keys:
        input_ids      [B, T]
        attention_mask  [B, T]
        completion_mask [B, T]  — 1 for completion tokens, 0 elsewhere
        labels          [B]     — scalar degeneration label per sequence
    """

    def collate_fn(batch: List[DegenerationItem]) -> Dict[str, torch.Tensor]:
        prompts = [item.prompt for item in batch]
        completions = [item.completion for item in batch]
        labels = torch.tensor([item.label for item in batch], dtype=torch.float32)

        # Build full texts: prompt + completion
        # We tokenize the prompt alone first to find the boundary.
        prompt_encodings = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

        # Full sequence: prompt + completion
        full_texts = [p + c for p, c in zip(prompts, completions)]
        full_encodings = tokenizer(
            full_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = full_encodings["input_ids"]
        attention_mask = full_encodings["attention_mask"]

        # Build completion_mask: 1 for completion tokens, 0 for prompt/padding
        B, T = input_ids.shape
        completion_mask = torch.zeros(B, T, dtype=torch.float32)
        for i, plen in enumerate(prompt_lengths):
            # Completion tokens start after the prompt and end where attention_mask is 1
            seq_len = attention_mask[i].sum().item()
            if plen < seq_len:
                completion_mask[i, plen:seq_len] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "labels": labels,
        }

    return collate_fn
