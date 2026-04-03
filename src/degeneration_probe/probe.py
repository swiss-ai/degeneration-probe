"""SequenceProbe: linear probe on frozen LLM hidden states for degeneration detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from degeneration_probe.model_utils import get_model_layers, get_model_hidden_size
from degeneration_probe.utils.hooks import add_hooks


class SequenceProbe(nn.Module):
    """
    A lightweight probe that hooks into a frozen LLM at a given layer,
    pools the hidden states over completion tokens, and applies a linear
    head to produce a scalar degeneration logit per sequence.

    The base model is frozen — only the linear head is trained.
    """

    POOLING_MODES = ("mean", "max", "last_token")

    def __init__(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        pooling: str = "mean",
        seed: int = 42,
    ) -> None:
        super().__init__()

        if pooling not in self.POOLING_MODES:
            raise ValueError(f"pooling must be one of {self.POOLING_MODES}, got '{pooling}'")

        self.model = model
        self.pooling = pooling

        # Resolve layer
        layers = get_model_layers(model)
        num_layers = len(layers)
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if not 0 <= layer_idx < num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers})")
        self.layer_idx = layer_idx
        self.target_module = layers[layer_idx]

        # Linear head
        hidden_size = get_model_hidden_size(model)
        self.hidden_size = hidden_size
        torch.manual_seed(seed)
        self.linear = nn.Linear(hidden_size, 1, device=model.device, dtype=torch.float32)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

        # Freeze the base model
        for param in model.parameters():
            param.requires_grad = False

        # Hook state
        self._hooked: Optional[torch.Tensor] = None
        self._hook_fn = self._make_hook()

    def _make_hook(self):
        """Forward hook that captures hidden states from the target layer."""
        def hook_fn(module, module_input, module_output):
            # Most architectures return (hidden_states, ...) as a tuple
            if isinstance(module_output, tuple) and module_output[0].ndim == 3:
                self._hooked = module_output[0]
            else:
                self._hooked = module_output
        return hook_fn

    def _pool(
        self,
        hidden: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool hidden states over completion tokens.

        Args:
            hidden: [B, T, H] hidden states from the hooked layer.
            completion_mask: [B, T] binary mask (1 = completion token).

        Returns:
            Pooled representation [B, H].
        """
        mask = completion_mask.unsqueeze(-1)  # [B, T, 1]

        if self.pooling == "mean":
            denom = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            return (hidden * mask).sum(dim=1) / denom

        if self.pooling == "max":
            # Replace masked positions with -inf so they are never chosen
            filled = hidden.masked_fill(mask == 0, float("-inf"))
            return filled.max(dim=1).values

        # last_token: hidden state at the last completion token per sequence
        # Find the index of the last 1 in each row of completion_mask
        last_idx = completion_mask.cumsum(dim=1).argmax(dim=1)  # [B]
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: frozen LLM → hook → pool → linear → logit.

        Args:
            input_ids:       [B, T]
            attention_mask:  [B, T]
            completion_mask: [B, T]

        Returns:
            Dict with "logit" [B, 1].
        """
        self._hooked = None
        fwd_hooks = [(self.target_module, self._hook_fn)]

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

        if self._hooked is None:
            raise RuntimeError("Failed to capture hidden states from target layer")

        hidden = self._hooked  # [B, T, H]

        # Cast to linear head dtype if needed
        if hidden.dtype != self.linear.weight.dtype:
            hidden = hidden.to(self.linear.weight.dtype)

        pooled = self._pool(hidden, completion_mask.to(hidden.device))  # [B, H]
        logit = self.linear(pooled)  # [B, 1]

        return {"logit": logit}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save probe head weights and config to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.linear.state_dict(), path / "probe_head.pt")

        config = {
            "layer_idx": self.layer_idx,
            "hidden_size": self.hidden_size,
            "pooling": self.pooling,
        }
        with open(path / "probe_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: PreTrainedModel,
    ) -> "SequenceProbe":
        """Load a saved probe head onto *model*."""
        path = Path(path)
        with open(path / "probe_config.json") as f:
            config = json.load(f)

        probe = cls(
            model=model,
            layer_idx=config["layer_idx"],
            pooling=config["pooling"],
        )

        state_dict = torch.load(
            path / "probe_head.pt",
            map_location=model.device,
            weights_only=True,
        )
        probe.linear.load_state_dict(state_dict)

        return probe
