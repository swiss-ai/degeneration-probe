"""Render per-token HTML heatmaps for a single prompt across MULTIPLE probe
checkpoints (one per layer in a sweep). Produces one HTML per sample with the
deterministic ground-truth row plus one row per layer probe, all aligned on the
same completion tokens and sharing the [0, 1] colour scale.

Loads the base model once and swaps the LoRA adapter + probe head per layer.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.data.converters import prepare_repetition_instruct_token_level_dataset
from degeneration_probe.data.dataset import (
    TokenizedProbingDataset,
    TokenizedProbingDatasetConfig,
)
from degeneration_probe.probes.value_head_probe import ValueHeadProbe
from degeneration_probe.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype


ID_CANDIDATE_FIELDS = ("prompt_id", "id", "sample_id", "uuid", "name", "example_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dirs", nargs="+", required=True,
                   help="One or more probe checkpoint dirs (one per layer).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--hf-repo", default="lorenzo0312/degeneration-probe-instruct-token-level-balanced")
    p.add_argument("--split", default="validation")
    p.add_argument("--subset", default=None)
    p.add_argument("--model-name", default="swiss-ai/Apertus-8B-Instruct-2509")
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--max-length", type=int, default=4608)
    p.add_argument("--max-completion-length", type=int, default=4096)
    return p.parse_args()


def detect_id_column(ds: datasets.Dataset, target_ids: List[str]) -> str:
    target_set = set(target_ids)
    for field in ID_CANDIDATE_FIELDS:
        if field in ds.column_names:
            values = ds[field]
            if any(v in target_set for v in values):
                return field
    for field in ds.column_names:
        values = ds[field]
        if not values:
            continue
        if not isinstance(values[0], str):
            continue
        if any(v in target_set for v in values):
            return field
    raise ValueError(
        f"Could not locate ids {target_ids} in any column. Columns: {ds.column_names}"
    )


def load_target_rows(hf_repo: str, split: str, subset: Optional[str],
                     sample_ids: List[str]) -> Tuple[datasets.Dataset, str]:
    print(f"[viz-layers] Loading {hf_repo} ({subset=}, {split=})")
    if subset:
        ds = datasets.load_dataset(hf_repo, subset, split=split)
    else:
        ds = datasets.load_dataset(hf_repo, split=split)
    id_col = detect_id_column(ds, sample_ids)
    print(f"[viz-layers] Using '{id_col}' as the sample-id column")
    id_to_index = {v: i for i, v in enumerate(ds[id_col]) if v in set(sample_ids)}
    missing = [sid for sid in sample_ids if sid not in id_to_index]
    if missing:
        raise ValueError(f"Sample ids not found in split={split}: {missing}")
    indices = [id_to_index[sid] for sid in sample_ids]
    return ds.select(indices), id_col


def build_dataset_for_rows(rows, tokenizer, *, max_length, max_completion_length) -> TokenizedProbingDataset:
    items = prepare_repetition_instruct_token_level_dataset(rows)
    cfg = TokenizedProbingDatasetConfig(
        dataset_id="viz_subset",
        hf_repo="(local-subset)",
        split="(local)",
        max_length=max_length,
        max_completion_length=max_completion_length,
        prompt_truncation_side="left",
        default_ignore=True,
        last_span_token=False,
        ignore_buffer=0,
        pos_weight=1.0,
        neg_weight=1.0,
        shuffle=False,
        seed=42,
        process_on_the_fly=False,
        max_num_samples=None,
    )
    return TokenizedProbingDataset(items=items, config=cfg, tokenizer=tokenizer)


@torch.no_grad()
def score_sample(probe: ValueHeadProbe, item: Dict[str, torch.Tensor]) -> np.ndarray:
    input_ids = item["input_ids"].unsqueeze(0).to(probe.device)
    attention_mask = item["attention_mask"].unsqueeze(0).to(probe.device)
    outputs = probe(input_ids=input_ids, attention_mask=attention_mask)
    probe_logits = outputs["probe_logits"].squeeze(-1).squeeze(0)
    return torch.sigmoid(probe_logits.float()).cpu().numpy()


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

def _score_to_rgb(score: float) -> Tuple[int, int, int]:
    s = max(0.0, min(1.0, float(score)))
    r = int(round(255 * s))
    g = int(round(255 * (1.0 - s)))
    return (r, g, 0)


def _escape_token(tok: str) -> str:
    out = html.escape(tok)
    out = out.replace("\n", "<span class='nl'>↵</span><br>")
    out = out.replace("\t", "<span class='ws'>→</span>")
    return out


LEGEND_HTML = """
<div class="legend">
  <span class="legend-label">repetition score</span>
  <div class="legend-bar"></div>
  <div class="legend-ticks">
    <span>0.00</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span>
  </div>
  <span class="legend-note">Tokens with no deterministic score are rendered in light grey.</span>
</div>
"""

STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 24px; color: #111; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  h2 { font-size: 13px; margin-top: 22px; margin-bottom: 6px; color: #444; }
  .meta { color: #555; font-size: 12px; margin-bottom: 16px; }
  .prompt { background: #f6f6f6; padding: 12px 14px; border-left: 3px solid #888;
            font-family: ui-monospace, Menlo, Monaco, monospace; font-size: 12px;
            white-space: pre-wrap; max-height: 220px; overflow-y: auto; }
  .completion { font-family: ui-monospace, Menlo, Monaco, monospace; font-size: 12.5px;
                line-height: 1.6; padding: 12px; border: 1px solid #ddd;
                border-radius: 4px; background: #fff; word-wrap: break-word; }
  .tok { padding: 1px 2px; border-radius: 2px; margin: 0 1px; color: #111; }
  .tok.ignored { background: #e5e5e5; color: #555; }
  .nl, .ws { color: #aaa; }
  .row-tag { display: inline-block; font-family: ui-monospace, Menlo, Monaco, monospace;
             font-size: 11px; color: #888; margin-right: 8px; }
  .legend { display: inline-block; margin-top: 20px; }
  .legend-label { display: block; font-size: 12px; color: #444; margin-bottom: 4px; }
  .legend-bar { width: 300px; height: 14px; border: 1px solid #aaa;
                background: linear-gradient(90deg,
                   rgb(0,255,0) 0%,
                   rgb(64,191,0) 25%,
                   rgb(128,128,0) 50%,
                   rgb(191,64,0) 75%,
                   rgb(255,0,0) 100%); }
  .legend-ticks { display: flex; justify-content: space-between; width: 300px;
                  font-size: 11px; color: #555; margin-top: 2px; }
  .legend-note { display: block; font-size: 11px; color: #777; margin-top: 6px; }
</style>
"""


def _render_token_row(completion_tokens: List[str],
                      scores: List[Optional[float]]) -> str:
    spans = []
    for tok, score in zip(completion_tokens, scores):
        tok_html = _escape_token(tok)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            spans.append(f"<span class='tok ignored' title='no score'>{tok_html}</span>")
        else:
            r, g, b = _score_to_rgb(score)
            spans.append(
                f"<span class='tok' style='background:rgb({r},{g},{b})' "
                f"title='{score:.3f}'>{tok_html}</span>"
            )
    return f"<div class='completion'>{''.join(spans)}</div>"


def render_multilayer_html(
    *,
    sample_id: str,
    prompt_text: str,
    completion_tokens: List[str],
    deterministic_scores: List[Optional[float]],
    per_layer_scores: List[Tuple[int, List[float]]],  # sorted (layer_idx, scores)
) -> str:
    parts = [
        f"<h1>{html.escape(sample_id)} — probe score by layer</h1>",
        "<div class='meta'>One row per layer. Color scale fixed to [0, 1]. "
        "Hover a token to see the score.</div>",
        f"<h2>Prompt</h2><div class='prompt'>{html.escape(prompt_text)}</div>",
        "<h2>Ground truth (deterministic checker)</h2>",
        _render_token_row(completion_tokens, deterministic_scores),
    ]
    for layer_idx, scores in per_layer_scores:
        parts.append(f"<h2>Probe — layer {layer_idx}</h2>")
        parts.append(_render_token_row(completion_tokens, scores))
    parts.append(LEGEND_HTML)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"{STYLE}</head><body>{''.join(parts)}</body></html>"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, id_col = load_target_rows(args.hf_repo, args.split, args.subset, args.sample_ids)
    sample_ids_in_order = list(rows[id_col])

    print(f"[viz-layers] Loading base model {args.model_name} (dtype={args.model_dtype})")
    base_model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        torch_dtype=resolve_torch_dtype(args.model_dtype),
    )
    tokenizer.padding_side = "right"
    for _, param in base_model.named_parameters():
        param.requires_grad = False

    viz_ds = build_dataset_for_rows(
        rows,
        tokenizer,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
    )

    # Cache per-sample static data (tokens, prompt text, deterministic scores).
    samples: Dict[str, Dict] = {}
    for sample_idx, sample_id in enumerate(sample_ids_in_order):
        item = viz_ds[sample_idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["classification_labels"]
        lm_labels = item["lm_labels"]
        completion_mask = (lm_labels != -100) & (attention_mask == 1)
        completion_idx = torch.nonzero(completion_mask, as_tuple=False).flatten().tolist()
        if not completion_idx:
            print(f"[viz-layers] No completion tokens for {sample_id}, skipping")
            continue
        prompt_mask = (attention_mask == 1) & (~completion_mask)
        prompt_text = tokenizer.decode(
            input_ids[prompt_mask].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        completion_tokens = [
            tokenizer.decode([int(input_ids[i])], clean_up_tokenization_spaces=False)
            for i in completion_idx
        ]
        det_full = labels.cpu().numpy()
        det_scores = [
            (None if det_full[i] == -100.0 else float(det_full[i]))
            for i in completion_idx
        ]
        samples[sample_id] = {
            "item": item,
            "completion_idx": completion_idx,
            "prompt_text": prompt_text,
            "completion_tokens": completion_tokens,
            "det_scores": det_scores,
        }

    # Probe loop: load adapter + head, score every sample, unload.
    per_layer_scores: Dict[str, Dict[int, List[float]]] = {sid: {} for sid in samples}
    layer_indices: List[int] = []
    probe_dirs = [Path(p) for p in args.probe_dirs]

    for pdir in probe_dirs:
        cfg = json.load(open(pdir / "probe_config.json"))
        layer_idx = int(cfg["layer_idx"])
        layer_indices.append(layer_idx)
        print(f"[viz-layers] === layer {layer_idx} from {pdir} ===")

        if (pdir / "adapter_config.json").exists():
            print(f"[viz-layers] Attaching LoRA adapter")
            model = PeftModel.from_pretrained(base_model, str(pdir))
        else:
            model = base_model
        probe = ValueHeadProbe(model, path=pdir)
        probe.eval()

        for sample_id, s in samples.items():
            probe_scores_full = score_sample(probe, s["item"])
            per_layer_scores[sample_id][layer_idx] = [
                float(probe_scores_full[i]) for i in s["completion_idx"]
            ]

        del probe
        if isinstance(model, PeftModel):
            base_model = model.unload()  # restore unwrapped base
        del model
        torch.cuda.empty_cache()

    layer_indices_sorted = sorted(set(layer_indices))

    for sample_id, s in samples.items():
        html_str = render_multilayer_html(
            sample_id=sample_id,
            prompt_text=s["prompt_text"],
            completion_tokens=s["completion_tokens"],
            deterministic_scores=s["det_scores"],
            per_layer_scores=[(l, per_layer_scores[sample_id][l]) for l in layer_indices_sorted],
        )
        out_path = args.output_dir / f"{sample_id}_layers.html"
        out_path.write_text(html_str)
        print(f"[viz-layers] wrote {out_path}")

        npz_arrays = {
            "completion_token_indices": np.array(s["completion_idx"], dtype=np.int64),
            "completion_tokens": np.array(s["completion_tokens"], dtype=object),
            "deterministic": np.array([sc if sc is not None else np.nan for sc in s["det_scores"]]),
        }
        for l in layer_indices_sorted:
            npz_arrays[f"probe_layer_{l}"] = np.array(per_layer_scores[sample_id][l])
        np.savez(args.output_dir / f"{sample_id}_layers_scores.npz", **npz_arrays)

    summary = {
        "probe_dirs": [str(p) for p in probe_dirs],
        "layer_indices": layer_indices_sorted,
        "hf_repo": args.hf_repo,
        "split": args.split,
        "subset": args.subset,
        "id_column": id_col,
        "sample_ids": sample_ids_in_order,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[viz-layers] done — files in {args.output_dir}")


if __name__ == "__main__":
    main()
