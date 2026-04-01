"""Minimal model-loading utilities copied and reduced from feature-probes."""

from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(
    dtype_name: Optional[str],
    *,
    default: Optional[torch.dtype] = None,
) -> Optional[torch.dtype]:
    """Resolve config dtype name to a torch dtype."""
    if dtype_name is None:
        return default

    key = str(dtype_name).strip().lower()
    if key in {"", "auto", "default", "model"}:
        return default
    if key in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if key in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16

    raise ValueError(
        f"Unsupported dtype '{dtype_name}'. Use one of: auto, float32, float16, bfloat16."
    )


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str,
    device_map: Optional[Union[str, dict]] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Hugging Face causal LM and its tokenizer."""
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
