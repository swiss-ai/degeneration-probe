"""Utilities for loading Hugging Face causal language models."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


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
        elif torch.backends.mps.is_available():
            # MPS does not support bfloat16 matmul with mixed accumulators
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=torch_dtype,
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


def get_model_layers(model: PreTrainedModel) -> List[nn.Module]:
    """Get the list of transformer layers from a model."""
    # Common patterns for accessing layers in different model architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA, Mistral, Qwen, etc.
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2, GPT-J, etc.
        return list(model.transformer.h)
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        # GPT-NeoX
        return list(model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def get_num_layers(model_or_name: Union[str, PreTrainedModel]) -> int:
    """Get the number of transformer layers in a model."""
    if isinstance(model_or_name, str):
        model_layers_map = {
            "swiss-ai/Apertus-8B-Instruct-2509": 32,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 80,
            "google/gemma-2-2b-it": 26,
            "google/gemma-2-9b-it": 42,
            "Qwen/Qwen2.5-0.5B-Instruct": 24,
            "Qwen/Qwen2.5-1.5B-Instruct": 28,
            "Qwen/Qwen2.5-3B-Instruct": 36,
            "Qwen/Qwen2.5-7B-Instruct": 28,
            "Qwen/Qwen2.5-14B-Instruct": 48,
            "mistralai/Mistral-Small-24B-Instruct-2501": 40,
        }
        if model_or_name in model_layers_map:
            return model_layers_map[model_or_name]
        raise ValueError(
            f"Model '{model_or_name}' not in layer map. "
            f"Pass a loaded model instance or add it to the map."
        )
    return len(get_model_layers(model_or_name))


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """Get the hidden size of a transformer model."""
    if hasattr(model, 'config'):
        for attr in ('hidden_size', 'd_model', 'n_embd', 'embed_dim'):
            if hasattr(model.config, attr):
                return getattr(model.config, attr)

    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.shape[1]

    raise ValueError(f"Could not determine hidden size for {type(model)}")
