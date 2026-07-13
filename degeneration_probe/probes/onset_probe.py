"""Standalone probe heads for the no-LoRA onset probes (`probe_N` / multi-horizon).

Deliberately NOT `ValueHeadProbe` -- that class hooks a live model's forward pass
and belongs to the live-LoRA path (see the implementation prompt's "Existing code
to reuse" section: LoRA needs a live forward pass since cached activations become
invalid once the model's weights change). Here the "trunk" is just the frozen,
cached hidden state read directly off disk by `OnsetActivationDataset`, so the
probe is only ever this small head: a plain linear layer, or (if `mlp_hidden_dim`
is set) a 2-layer MLP with a ReLU in between.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class OnsetProbeHead(nn.Module):
    """`n_outputs=1` for a single `probe_N` head; `n_outputs=len(horizons)` for the
    multi-horizon head (one shared head, K logits per token)."""

    def __init__(self, hidden_size: int, n_outputs: int, mlp_hidden_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs
        self.mlp_hidden_dim = mlp_hidden_dim
        if mlp_hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, n_outputs),
            )
        else:
            self.net = nn.Linear(hidden_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "probe_head.bin")
        config = {
            "hidden_size": self.hidden_size,
            "n_outputs": self.n_outputs,
            "mlp_hidden_dim": self.mlp_hidden_dim,
        }
        with open(path / "probe_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "OnsetProbeHead":
        path = Path(path)
        with open(path / "probe_config.json") as f:
            config = json.load(f)
        probe = cls(config["hidden_size"], config["n_outputs"], config.get("mlp_hidden_dim"))
        state_dict = torch.load(path / "probe_head.bin", map_location=map_location, weights_only=True)
        probe.load_state_dict(state_dict)
        return probe
