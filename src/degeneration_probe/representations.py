"""Hidden states extraction for probe training (placeholder)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def extract_hidden_states(
    model,
    tokenizer,
    input_ids: Optional[torch.Tensor],
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Extract and optionally save the hidden states produced by ``model``.

    This function is a **placeholder**. Once implemented it will:

    1. Run a forward pass with ``output_hidden_states=True``.
    2. Stack the per-layer hidden states into a single tensor of shape
       ``(num_layers, seq_len, hidden_dim)``.
    3. Save the tensor to ``output_path`` (a ``.pt`` file) if provided.
    4. Return a dict with keys ``hidden_states_path``, ``num_layers``,
       ``hidden_dim`` that gets merged into the generation record.

    Currently returns ``{}`` so the rest of the pipeline runs unchanged
    while this module is under development.
    """
    return {}
