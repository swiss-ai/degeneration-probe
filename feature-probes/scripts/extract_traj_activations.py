"""Extract per-token Apertus-8B activations on a delayed-degeneration rollout.

Picks a known rollout (default: train row 612 -- Taylor-theorem limit, 613 clean tokens
then ~48 degenerate tokens). Runs a single teacher-forcing forward pass with
`output_hidden_states=True`, captures residual-stream activations at the requested
transformer layers for every generated-token position, and saves them with per-token
labels (degenerating bool, repetition score, decoded token string).

The script can be run twice on the same prompt:
    - base model only
    - base model + LoRA adapter loaded from --lora-path
to compare how the LoRA adaptation reshapes activations.

Output: a single .npz at <out_dir>/<out_name>.npz containing one array per requested
layer plus metadata.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_LAYERS = [1, 2, 5, 10, 26, 30]
DEFAULT_ARROW = (
    "/capstor/scratch/cscs/lbaggi/feature-probes/hf_cache/datasets/"
    "lorenzo0312___degeneration-probe-instruct-token-level-balanced/default/0.0.0/"
    "55831e27b7609554a4f3d40a3a6f18e481f24d4c/"
    "degeneration-probe-instruct-token-level-balanced-train.arrow"
)


def load_row(arrow_path: str, row_idx: int) -> dict:
    with pa.memory_map(arrow_path, "r") as source:
        table = ipc.open_stream(source).read_all()
    return table.slice(row_idx, 1).to_pylist()[0]


def build_prompt_text(tokenizer, prompt: str) -> str:
    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and tokenizer.bos_token in text:
        text = text.replace(tokenizer.bos_token, "")
    return text


def tokenize_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=612)
    parser.add_argument("--arrow", default=DEFAULT_ARROW)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Transformer layer indices (0-based) to dump.",
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Optional path to a PEFT/LoRA adapter directory to load on top of the base model.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--out-name",
        required=True,
        help="Output filename prefix (without extension). Final file: <out_name>.npz",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = load_row(args.arrow, args.row)
    print(
        f"Row {args.row}  source={row['source_dataset']}  "
        f"degenerating={row['degenerating']}  n_chunks={len(row['chunk_summary'])}",
        flush=True,
    )
    print(f"Prompt ({len(row['prompt'])} chars): {row['prompt'][:200]!r}", flush=True)

    print(f"Loading tokenizer + base model {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path} ...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()
        variant = "lora"
    else:
        variant = "base"

    n_hidden = (
        model.get_base_model().config.num_hidden_layers
        if isinstance(model, PeftModel)
        else model.config.num_hidden_layers
    )
    hidden_size = (
        model.get_base_model().config.hidden_size
        if isinstance(model, PeftModel)
        else model.config.hidden_size
    )
    print(f"variant={variant}  num_hidden_layers={n_hidden}  hidden_size={hidden_size}", flush=True)
    assert max(args.layers) < n_hidden, (
        f"requested layer {max(args.layers)} out of range for model with {n_hidden} layers"
    )

    prompt_text = build_prompt_text(tokenizer, row["prompt"])
    prompt_ids = tokenize_no_special(tokenizer, prompt_text)
    completion_ids = tokenize_no_special(tokenizer, row["generated_text"])
    input_ids = prompt_ids + completion_ids
    prompt_len = len(prompt_ids)
    completion_len = len(completion_ids)
    seq_len = len(input_ids)
    n_chunks = len(row["chunk_summary"])
    print(
        f"prompt_len={prompt_len}  completion_len={completion_len}  "
        f"seq_len={seq_len}  n_chunks={n_chunks}",
        flush=True,
    )
    if completion_len != n_chunks:
        print(
            f"WARNING: completion_len ({completion_len}) != n_chunks ({n_chunks}) -- "
            "labels will be aligned to min(completion_len, n_chunks)",
            flush=True,
        )

    n_tokens_labeled = min(completion_len, n_chunks)
    chunk_summary = row["chunk_summary"][:n_tokens_labeled]
    degenerating = np.array([c["degenerating"] for c in chunk_summary], dtype=bool)
    repetition = np.array([c["repetition"] for c in chunk_summary], dtype=np.float32)
    decoded_tokens = [
        tokenizer.decode([t], clean_up_tokenization_spaces=False)
        for t in completion_ids[:n_tokens_labeled]
    ]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=model.device)

    print("Forward pass (output_hidden_states=True, use_cache=False) ...", flush=True)
    with torch.no_grad():
        out = model(input_ids_t, output_hidden_states=True, use_cache=False)

    # hidden_states[0]  -> embedding output
    # hidden_states[i+1] -> output of decoder block i  (i in 0..n_hidden-1)
    hidden_states = out.hidden_states
    print(
        f"Got {len(hidden_states)} hidden states  per-state shape={tuple(hidden_states[0].shape)}",
        flush=True,
    )

    completion_slice = slice(prompt_len, prompt_len + n_tokens_labeled)

    activations = {}
    for layer in args.layers:
        hs = hidden_states[layer + 1][0, completion_slice].to(torch.float32).cpu().numpy()
        activations[f"layer_{layer}"] = hs
        print(
            f"  layer {layer:2d}: shape={hs.shape}  "
            f"norm_mean={np.linalg.norm(hs, axis=1).mean():.2f}",
            flush=True,
        )

    out_path = out_dir / f"{args.out_name}.npz"
    np.savez_compressed(
        out_path,
        layers=np.array(args.layers, dtype=np.int64),
        degenerating=degenerating,
        repetition=repetition,
        token_position=np.arange(n_tokens_labeled, dtype=np.int64),
        decoded_tokens=np.array(decoded_tokens, dtype=object),
        prompt=row["prompt"],
        generated_text=row["generated_text"],
        source_dataset=row["source_dataset"],
        row_index=args.row,
        prompt_len=prompt_len,
        completion_len=completion_len,
        variant=variant,
        lora_path=args.lora_path or "",
        **activations,
    )
    print(f"Saved -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
