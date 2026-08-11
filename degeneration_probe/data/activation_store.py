"""Reading one layer's cached activations for a rollout.

The cache holds every layer of every rollout, written once with the language
model frozen. Two details make direct indexing into it error prone, and both
are handled here so that no caller has to remember them:

- The stored tensor has one slot more than the model has layers. Slot 0 is the
  embedding output and slot ``i + 1`` is decoder block ``i``. The probe, by
  contrast, hooks a decoder block directly, so probe layer ``L`` lives at
  cached slot ``L + 1``. Off by one here is silent: the wrong layer trains
  perfectly well and simply answers a different question.
- The cache is addressed by rollout identity, never by a stored file path.
  Paths recorded at write time stop being true as soon as a build directory is
  renamed.

Every file carries the layer ordering in its own metadata, so the convention is
checked on read rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from safetensors import safe_open

# Slot 0 holds the embedding output, so decoder block L is at slot L + 1.
EMBEDDING_SLOTS = 1
LAYER_ORDER_LABEL = "embedding_then_blocks_0..31"
TENSOR_KEY = "hidden_states"


def cached_slot_for_probe_layer(probe_layer: int) -> int:
    """Translate a probed decoder block into its slot in the cached tensor."""
    if probe_layer < 0:
        raise ValueError(f"probe layer must be non-negative, got {probe_layer}")
    return probe_layer + EMBEDDING_SLOTS


def probe_layer_for_cached_slot(slot: int) -> int:
    """The inverse, for reading a cache index back into a layer number."""
    if slot < EMBEDDING_SLOTS:
        raise ValueError(f"cached slot {slot} is the embedding output, not a decoder block")
    return slot - EMBEDDING_SLOTS


def activation_path(
    build_root: Union[str, Path], domain: str, prompt_id: str, rollout_idx: int
) -> Path:
    """Rebuild a rollout's location from its identity, not from a stored path."""
    return (
        Path(build_root)
        / "activations"
        / str(domain)
        / str(prompt_id)
        / f"rollout_{int(rollout_idx)}.safetensors"
    )


def load_probe_layer(
    build_root: Union[str, Path],
    domain: str,
    prompt_id: str,
    rollout_idx: int,
    *,
    probe_layer: int,
    expected_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Return ``[num_tokens, hidden_size]`` for one probed layer of one rollout."""
    return load_probe_layers(
        build_root,
        domain,
        prompt_id,
        rollout_idx,
        probe_layers=[probe_layer],
        expected_tokens=expected_tokens,
    )[0]


def load_probe_layers(
    build_root: Union[str, Path],
    domain: str,
    prompt_id: str,
    rollout_idx: int,
    *,
    probe_layers: Sequence[int],
    expected_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Return ``[len(probe_layers), num_tokens, hidden_size]`` from one open.

    Opening the file and seeking dominates the cost of reading it, so asking for
    every layer costs little more than asking for one. That is what makes a
    probe per layer affordable: the depths can be trained together in a single
    pass over the data rather than one pass each.
    """
    path = activation_path(build_root, domain, prompt_id, rollout_idx)
    if not path.is_file():
        raise FileNotFoundError(f"No cached activations for {domain}/{prompt_id}/{rollout_idx}: {path}")
    requested = [int(layer) for layer in probe_layers]
    if not requested:
        raise ValueError("at least one probe layer is required")
    slots_wanted = [cached_slot_for_probe_layer(layer) for layer in requested]
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        recorded = metadata.get("layer_order")
        if recorded is not None and recorded != LAYER_ORDER_LABEL:
            raise ValueError(
                f"{path} declares layer order {recorded!r}, expected {LAYER_ORDER_LABEL!r}; "
                "the slot of a probed layer cannot be assumed under a different ordering"
            )
        sliced = handle.get_slice(TENSOR_KEY)
        slots, num_tokens, _ = sliced.get_shape()
        for layer, slot in zip(requested, slots_wanted):
            if slot >= slots:
                raise ValueError(
                    f"probe layer {layer} maps to cached slot {slot}, but {path} holds "
                    f"{slots} slots ({slots - EMBEDDING_SLOTS} decoder blocks)"
                )
        if expected_tokens is not None and num_tokens != expected_tokens:
            raise ValueError(
                f"{path} holds {num_tokens} tokens, but the rollout is {expected_tokens} tokens long"
            )
        # One contiguous read spanning the requested slots, then the wanted rows
        # out of it. Seeking to each slot separately is what costs, not the
        # bytes, so a span plus a gather beats a slice per layer. Narrowing the
        # token range instead would be slower still, since that turns one
        # sequential read into a strided one per layer.
        low, high = min(slots_wanted), max(slots_wanted)
        span = sliced[low : high + 1]
        wanted = [slot - low for slot in slots_wanted]
        # The gather copies the whole span, which for every layer of a long
        # rollout is hundreds of megabytes. Asking for the span itself is the
        # common case and needs no copy at all.
        if wanted == list(range(span.shape[0])):
            return span
        return span[wanted]


def agrees_with_live_capture(
    cached: torch.Tensor, captured: torch.Tensor, *, tolerance: float = 5e-3
) -> bool:
    """Whether a cached layer matches what a live forward pass produces.

    Worth running once whenever the cache, the model or the probed layer
    changes: it is the only check that catches a slot mistake, since training on
    the wrong layer produces perfectly plausible numbers. The tolerance is loose
    because the cache is float16 while a live capture is usually bfloat16.
    """
    if cached.shape != captured.shape:
        return False
    return torch.allclose(cached.float(), captured.float(), atol=tolerance, rtol=tolerance)
