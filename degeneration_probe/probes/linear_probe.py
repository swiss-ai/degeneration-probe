"""A scalar linear per-token probe, over live or cached hidden states.

The trained part is the same in both feature regimes: a normalization and one
linear layer mapping a token's hidden state to a single logit. What differs is
where the hidden state comes from. With the language model in the loop it is
captured by a hook during a forward pass, which is what adapters require since
they change the representation at every step. With the model frozen the same
vectors are constants and can be read from the activation cache, which removes
the language model from training entirely.

The two share their head logic and their checkpoint layout, so a probe trained
on cached activations loads unchanged into a live model and the reverse. That
equivalence is worth a test rather than an assumption: divergence here would be
invisible, since both would keep producing plausible numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

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

PROBE_WEIGHTS = "probe.bin"
NORM_WEIGHTS = "normalization.bin"
PROBE_CONFIG = "probe_config.json"


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / inputs.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)


def build_normalization(
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


def context_window_features(states: torch.Tensor, context_window_size: int) -> torch.Tensor:
    """Concatenate each token's state with the states just before it."""
    if context_window_size == 1:
        return states
    shifted = [states]
    for offset in range(1, context_window_size):
        previous = states.roll(offset, dims=1)
        previous = previous.clone()
        previous[:, :offset, :] = 0
        shifted.append(previous)
    return torch.cat(shifted, dim=-1)


def apply_head(
    states: torch.Tensor,
    *,
    pre_head_norm: nn.Module,
    linear: nn.Module,
    context_window_size: int,
) -> torch.Tensor:
    """Map ``[batch, tokens, hidden]`` to one logit per token."""
    features = context_window_features(states, context_window_size)
    norm_parameter = next(pre_head_norm.parameters(), None)
    if norm_parameter is not None:
        features = features.to(norm_parameter.dtype)
    features = pre_head_norm(features).to(linear.weight.dtype)
    return linear(features).squeeze(-1)


def _init_head(
    owner: nn.Module,
    *,
    hidden_size: int,
    context_window_size: int,
    probe_dtype: str,
    normalization: str,
    seed: int,
    device: torch.device,
    default_dtype: torch.dtype,
) -> None:
    input_size = hidden_size * context_window_size
    head_dtype = resolve_torch_dtype(probe_dtype, default=default_dtype)
    owner.pre_head_norm = build_normalization(
        normalization, input_size, device=device, dtype=head_dtype
    )
    torch.manual_seed(seed)
    owner.linear = nn.Linear(input_size, 1, device=device, dtype=head_dtype)
    nn.init.normal_(owner.linear.weight, mean=0.0, std=0.01)
    nn.init.zeros_(owner.linear.bias)


def _save_head(owner: nn.Module, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(owner.linear.state_dict(), path / PROBE_WEIGHTS)
    if owner.pre_head_norm.state_dict():
        torch.save(owner.pre_head_norm.state_dict(), path / NORM_WEIGHTS)
    config = {
        "architecture": "scalar_linear",
        "layer_idx": owner.layer_idx,
        "context_window_size": owner.context_window_size,
        "probe_dtype": owner.probe_dtype,
        "normalization": owner.normalization,
        "input_size": owner.linear.in_features,
        "output_size": owner.linear.out_features,
    }
    with (path / PROBE_CONFIG).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _load_head(owner: nn.Module, path: Path) -> None:
    owner.linear.load_state_dict(
        torch.load(path / PROBE_WEIGHTS, map_location=owner.device, weights_only=True)
    )
    norm_path = path / NORM_WEIGHTS
    if norm_path.is_file() and owner.pre_head_norm.state_dict():
        owner.pre_head_norm.load_state_dict(
            torch.load(norm_path, map_location=owner.device, weights_only=True)
        )


def read_probe_config(path: Union[str, Path]) -> Dict[str, object]:
    with (Path(path) / PROBE_CONFIG).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class DegenerationProbe(nn.Module):
    """Capture token states from a live model and map each to one logit."""

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

        if path is not None:
            path = Path(path)
            checkpoint_config = read_probe_config(path)
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
        _init_head(
            self,
            hidden_size=get_model_hidden_size(model),
            context_window_size=self.context_window_size,
            probe_dtype=probe_dtype,
            normalization=normalization,
            seed=seed,
            device=self.device,
            default_dtype=getattr(model, "dtype", next(model.parameters()).dtype),
        )
        if path is not None:
            _load_head(self, path)

        self._captured: Optional[torch.Tensor] = None

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
        logits = apply_head(
            self._captured,
            pre_head_norm=self.pre_head_norm,
            linear=self.linear,
            context_window_size=self.context_window_size,
        )
        return {"probe_logits": logits}

    @property
    def device(self) -> torch.device:
        return getattr(self.model, "device", next(self.model.parameters()).device)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Restore head and adapter weights in place, for resuming a run."""
        path = Path(path)
        _load_head(self, path)
        if isinstance(self.model, PeftModel):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter = path / "adapter_model.safetensors"
            if adapter.is_file():
                set_peft_model_state_dict(self.model, load_file(str(adapter)))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        _save_head(self, path)


class CachedFeatureProbe(nn.Module):
    """The same head, reading hidden states that were computed once already.

    With the language model frozen its hidden states are constants, so training
    need not run it at all. The head is identical to the live one and writes the
    same checkpoint, so which regime produced a probe is not visible downstream.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_idx: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        context_window_size: int = 1,
        probe_dtype: str = "float32",
        normalization: str = "layernorm",
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if layer_idx is None and path is None:
            raise ValueError("Either layer_idx or path is required")
        if path is not None:
            path = Path(path)
            checkpoint_config = read_probe_config(path)
            saved_layer = int(checkpoint_config["layer_idx"])
            if layer_idx is not None and layer_idx != saved_layer:
                raise ValueError(f"Configured layer {layer_idx} differs from checkpoint layer {saved_layer}")
            layer_idx = saved_layer
            context_window_size = int(checkpoint_config["context_window_size"])
            probe_dtype = checkpoint_config["probe_dtype"]
            normalization = checkpoint_config["normalization"]

        self.layer_idx = int(layer_idx)
        self.context_window_size = int(context_window_size)
        self.probe_dtype = probe_dtype
        self.normalization = normalization
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        _init_head(
            self,
            hidden_size=hidden_size,
            context_window_size=self.context_window_size,
            probe_dtype=probe_dtype,
            normalization=normalization,
            seed=seed,
            device=self._device,
            default_dtype=torch.float32,
        )
        if path is not None:
            _load_head(self, path)

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        logits = apply_head(
            features.to(self.linear.weight.dtype),
            pre_head_norm=self.pre_head_norm,
            linear=self.linear,
            context_window_size=self.context_window_size,
        )
        return {"probe_logits": logits}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_weights(self, path: Union[str, Path]) -> None:
        _load_head(self, Path(path))

    def save(self, path: Union[str, Path]) -> None:
        _save_head(self, Path(path))


class _Head(nn.Module):
    """One depth's normalization and linear map: the unit that gets trained."""

    def __init__(
        self,
        *,
        hidden_size: int,
        context_window_size: int,
        probe_dtype: str,
        normalization: str,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        _init_head(
            self,
            hidden_size=hidden_size,
            context_window_size=context_window_size,
            probe_dtype=probe_dtype,
            normalization=normalization,
            seed=seed,
            device=device,
            default_dtype=torch.float32,
        )


class MultiLayerCachedProbe(nn.Module):
    """One independent head per depth, trained together over one pass.

    The heads share nothing: each owns its normalization and its weights, and
    the training loss is their sum. A sum of independent terms has, for each
    head, exactly the gradient that head would receive on its own, so this is
    not an approximation of training the depths separately but a rearrangement
    of it. Two things would quietly break that equivalence and are handled
    elsewhere: a gradient norm clipped across all heads at once, and a shared
    normalization. Both are the reason the equivalence is asserted by a test.

    What makes it worth doing is the cache layout rather than arithmetic. Every
    depth of a rollout lives in one file, and opening that file costs far more
    than reading it, so the depths cost little more together than one costs
    alone.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_indices: Sequence[int],
        context_window_size: int = 1,
        probe_dtype: str = "float32",
        normalization: str = "layernorm",
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.layer_indices = [int(layer) for layer in layer_indices]
        if not self.layer_indices:
            raise ValueError("a multi-layer probe needs at least one layer")
        self.context_window_size = int(context_window_size)
        self.probe_dtype = probe_dtype
        self.normalization = normalization
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        # Seeded per depth, so a head's initialization does not depend on which
        # other depths happen to share the run with it.
        self.heads = nn.ModuleList(
            [
                _Head(
                    hidden_size=hidden_size,
                    context_window_size=self.context_window_size,
                    probe_dtype=probe_dtype,
                    normalization=normalization,
                    seed=seed,
                    device=self._device,
                )
                for _ in self.layer_indices
            ]
        )

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """``[batch, tokens, layers, hidden]`` to ``[batch, tokens, layers]``."""
        if features.shape[-2] != len(self.heads):
            raise ValueError(
                f"features carry {features.shape[-2]} layers but the probe has "
                f"{len(self.heads)} heads; the loader and the probe disagree about depth"
            )
        logits = [
            apply_head(
                features[..., index, :].to(head.linear.weight.dtype),
                pre_head_norm=head.pre_head_norm,
                linear=head.linear,
                context_window_size=self.context_window_size,
            )
            for index, head in enumerate(self.heads)
        ]
        return {"probe_logits": torch.stack(logits, dim=-1)}

    def head_parameters(self) -> list[list[nn.Parameter]]:
        """Each head's parameters on their own, for clipping them separately."""
        return [list(head.parameters()) for head in self.heads]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path: Union[str, Path]) -> None:
        """One directory per depth, each holding exactly a single-layer probe.

        Written this way so a head from a joint run is indistinguishable from
        one trained alone, and loads into the ordinary probe without a special
        case anywhere downstream.
        """
        path = Path(path)
        for layer, head in zip(self.layer_indices, self.heads):
            head.layer_idx = layer
            head.context_window_size = self.context_window_size
            head.probe_dtype = self.probe_dtype
            head.normalization = self.normalization
            _save_head(head, path / f"layer_{layer:02d}")

    def load_weights(self, path: Union[str, Path]) -> None:
        path = Path(path)
        for layer, head in zip(self.layer_indices, self.heads):
            _load_head(head, path / f"layer_{layer:02d}")


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


def setup_cached_probe(
    probe_config: ProbeConfig,
    *,
    hidden_size: int,
    checkpoint_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Union[CachedFeatureProbe, MultiLayerCachedProbe]:
    """Build the head alone, for training against the activation cache.

    Naming several layers builds one head per layer instead, trained together
    over a single pass of the cache.
    """
    if probe_config.layers is not None:
        if checkpoint_path is not None:
            probe = MultiLayerCachedProbe(
                hidden_size=hidden_size,
                layer_indices=probe_config.layers,
                context_window_size=probe_config.context_window_size,
                probe_dtype=probe_config.dtype,
                normalization=probe_config.normalization,
                seed=seed,
                device=device,
            )
            probe.load_weights(checkpoint_path)
            return probe
        return MultiLayerCachedProbe(
            hidden_size=hidden_size,
            layer_indices=probe_config.layers,
            context_window_size=probe_config.context_window_size,
            probe_dtype=probe_config.dtype,
            normalization=probe_config.normalization,
            seed=seed,
            device=device,
        )
    return CachedFeatureProbe(
        hidden_size=hidden_size,
        layer_idx=probe_config.layer,
        path=Path(checkpoint_path) if checkpoint_path is not None else None,
        context_window_size=probe_config.context_window_size,
        probe_dtype=probe_config.dtype,
        normalization=probe_config.normalization,
        seed=seed,
        device=device,
    )
