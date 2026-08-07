"""A scalar linear per-token probe attached to one transformer layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel

from degeneration_probe.config import LoraConfig, ProbeConfig
from degeneration_probe.utils.hooks import add_hooks
from degeneration_probe.utils.model_utils import (
    get_model_hidden_size,
    get_model_layers,
    resolve_torch_dtype,
    setup_lora_for_layers,
)


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / inputs.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)


class DegenerationProbe(nn.Module):
    """Capture token states and map each context vector to exactly one logit."""

    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        *,
        layer_idx: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        context_window_size: int = 1,
        probe_dtype: str = "float32",
        normalization: str = "layernorm",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if layer_idx is None and path is None:
            raise ValueError("Either layer_idx or path is required")

        checkpoint_config = None
        if path is not None:
            path = Path(path)
            with (path / "probe_config.json").open("r", encoding="utf-8") as handle:
                checkpoint_config = json.load(handle)
            saved_layer = int(checkpoint_config["layer_idx"])
            if layer_idx is not None and layer_idx != saved_layer:
                raise ValueError(f"Configured layer {layer_idx} differs from checkpoint layer {saved_layer}")
            layer_idx = saved_layer
            context_window_size = int(checkpoint_config["context_window_size"])
            probe_dtype = checkpoint_config["probe_dtype"]
            normalization = checkpoint_config["normalization"]

        self.model = model
        self.layer_idx = int(layer_idx)
        self.context_window_size = int(context_window_size)
        self.probe_dtype = probe_dtype
        self.normalization = normalization
        self.target_module = get_model_layers(model)[self.layer_idx]
        hidden_size = get_model_hidden_size(model)
        input_size = hidden_size * self.context_window_size
        model_dtype = getattr(model, "dtype", next(model.parameters()).dtype)
        head_dtype = resolve_torch_dtype(probe_dtype, default=model_dtype)
        self.pre_head_norm = self._build_norm(
            normalization, input_size, device=self.device, dtype=head_dtype
        )
        torch.manual_seed(seed)
        self.linear = nn.Linear(input_size, 1, device=self.device, dtype=head_dtype)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

        if path is not None:
            self.linear.load_state_dict(
                torch.load(path / "probe.bin", map_location=self.device, weights_only=True)
            )
            norm_path = path / "normalization.bin"
            if norm_path.is_file() and self.pre_head_norm.state_dict():
                self.pre_head_norm.load_state_dict(
                    torch.load(norm_path, map_location=self.device, weights_only=True)
                )

        self._captured: Optional[torch.Tensor] = None

    @staticmethod
    def _build_norm(
        kind: str, hidden_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> nn.Module:
        if kind == "none":
            return nn.Identity()
        if kind == "layernorm":
            return nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        if kind == "rmsnorm":
            return nn.RMSNorm(hidden_size, device=device, dtype=dtype)
        if kind == "l2":
            return L2Norm()
        raise ValueError(f"Unknown normalization {kind!r}")

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        self._captured = None
        with add_hooks(
            module_forward_pre_hooks=[],
            module_forward_hooks=[(self.target_module, self._capture)],
        ):
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                **kwargs,
            )
        if self._captured is None:
            raise RuntimeError("The configured transformer layer did not emit hidden states")

        shifted = [self._captured]
        for offset in range(1, self.context_window_size):
            previous = self._captured.roll(offset, dims=1)
            previous[:, :offset, :] = 0
            shifted.append(previous)
        features = torch.cat(shifted, dim=-1)
        norm_parameter = next(self.pre_head_norm.parameters(), None)
        if norm_parameter is not None:
            features = features.to(norm_parameter.dtype)
        features = self.pre_head_norm(features).to(self.linear.weight.dtype)
        return {"probe_logits": self.linear(features).squeeze(-1)}

    @property
    def device(self) -> torch.device:
        return getattr(self.model, "device", next(self.model.parameters()).device)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        torch.save(self.linear.state_dict(), path / "probe.bin")
        if self.pre_head_norm.state_dict():
            torch.save(self.pre_head_norm.state_dict(), path / "normalization.bin")
        config = {
            "architecture": "scalar_linear",
            "layer_idx": self.layer_idx,
            "context_window_size": self.context_window_size,
            "probe_dtype": self.probe_dtype,
            "normalization": self.normalization,
            "input_size": self.linear.in_features,
            "output_size": self.linear.out_features,
        }
        with (path / "probe_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)


def _lora_layers(config: LoraConfig, probe_layer: int) -> list[int]:
    if not config.enabled or config.layers == "none":
        return []
    if config.layers == "all":
        return list(range(probe_layer + 1))
    return [int(layer) for layer in config.layers]


def setup_probe(
    model: PreTrainedModel,
    probe_config: ProbeConfig,
    lora_config: LoraConfig,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> Tuple[PreTrainedModel, DegenerationProbe]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and (checkpoint / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    else:
        layers = _lora_layers(lora_config, probe_config.layer)
        if layers:
            model = setup_lora_for_layers(
                model,
                layers,
                lora_r=lora_config.rank,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
            )

    probe = DegenerationProbe(
        model,
        layer_idx=probe_config.layer,
        path=checkpoint,
        context_window_size=probe_config.context_window_size,
        probe_dtype=probe_config.dtype,
        normalization=probe_config.normalization,
        seed=seed,
    )
    return model, probe
