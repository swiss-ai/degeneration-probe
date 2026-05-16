"""Render per-token HTML heatmaps comparing deterministic repetition scores
against probe scores for a handful of dataset samples.

Two HTML files are produced per sample:
  - <id>_deterministic.html : ground-truth chunk_summary["repetition"] scores
  - <id>_probe.html         : sigmoid(probe_logits) from a trained ValueHeadProbe

All four plots share the same [0, 1] colour scale and embed the same colour
legend so they are directly comparable.
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
from transformers import PreTrainedModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_probes.data.converters import prepare_repetition_instruct_token_level_dataset
from feature_probes.data.dataset import (
    TokenizedProbingDataset,
    TokenizedProbingDatasetConfig,
)
from feature_probes.probes.value_head_probe import ValueHeadProbe
from feature_probes.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype


ID_CANDIDATE_FIELDS = ("prompt_id", "id", "sample_id", "uuid", "name", "example_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", type=Path, required=True,
                   help="Directory containing probe_head.bin / adapter_config.json / probe_config.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True,
                   help="Dataset sample ids to visualise")
    p.add_argument("--hf-repo", default="lorenzo0312/degeneration-probe-instruct-token-level-balanced")
    p.add_argument("--split", default="test")
    p.add_argument("--subset", default=None)
    p.add_argument("--model-name", default="swiss-ai/Apertus-8B-Instruct-2509")
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--max-length", type=int, default=4608)
    p.add_argument("--max-completion-length", type=int, default=4096)
    return p.parse_args()


def detect_id_column(ds: datasets.Dataset, target_ids: List[str]) -> str:
    """Return the column name that contains the given target ids."""
    target_set = set(target_ids)
    for field in ID_CANDIDATE_FIELDS:
        if field in ds.column_names:
            values = ds[field]
            if any(v in target_set for v in values):
                return field
    # Fall back: scan every string-typed column.
    for field in ds.column_names:
        values = ds[field]
        if not values:
            continue
        sample = values[0]
        if not isinstance(sample, str):
            continue
        if any(v in target_set for v in values):
            return field
    raise ValueError(
        f"Could not locate any of the requested ids {target_ids} in any column "
        f"of the dataset. Available columns: {ds.column_names}"
    )


def load_target_rows(
    hf_repo: str,
    split: str,
    subset: Optional[str],
    sample_ids: List[str],
) -> Tuple[datasets.Dataset, str]:
    print(f"[viz] Loading {hf_repo} ({subset=}, {split=})")
    if subset:
        ds = datasets.load_dataset(hf_repo, subset, split=split)
    else:
        ds = datasets.load_dataset(hf_repo, split=split)

    id_col = detect_id_column(ds, sample_ids)
    print(f"[viz] Using '{id_col}' as the sample-id column")

    id_to_index = {v: i for i, v in enumerate(ds[id_col]) if v in set(sample_ids)}
    missing = [sid for sid in sample_ids if sid not in id_to_index]
    if missing:
        raise ValueError(f"Sample ids not found in split={split}: {missing}")

    indices = [id_to_index[sid] for sid in sample_ids]
    return ds.select(indices), id_col


def build_dataset_for_rows(
    rows: datasets.Dataset,
    tokenizer,
    *,
    max_length: int,
    max_completion_length: int,
) -> TokenizedProbingDataset:
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


def load_probe(
    probe_dir: Path,
    model_name: str,
    model_dtype: str,
) -> Tuple[PreTrainedModel, ValueHeadProbe, "AutoTokenizer"]:
    print(f"[viz] Loading base model {model_name} (dtype={model_dtype})")
    base_model, tokenizer = load_model_and_tokenizer(
        model_name,
        torch_dtype=resolve_torch_dtype(model_dtype),
    )
    tokenizer.padding_side = "right"

    for _, param in base_model.named_parameters():
        param.requires_grad = False

    if (probe_dir / "adapter_config.json").exists():
        print(f"[viz] Attaching LoRA adapter from {probe_dir}")
        model = PeftModel.from_pretrained(base_model, str(probe_dir))
    else:
        model = base_model

    print(f"[viz] Loading probe head from {probe_dir}")
    probe = ValueHeadProbe(model, path=probe_dir)
    probe.eval()
    return model, probe, tokenizer


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
    """Make whitespace and newlines visible while keeping the text searchable."""
    out = html.escape(tok)
    out = out.replace("\n", "<span class='nl'>↵</span><br>")
    out = out.replace("\t", "<span class='ws'>→</span>")
    return out


LEGEND_HTML = """
<div class=\"legend\">
  <span class=\"legend-label\">repetition score</span>
  <div class=\"legend-bar\"></div>
  <div class=\"legend-ticks\">
    <span>0.00</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span>
  </div>
  <span class=\"legend-note\">Tokens with no deterministic score are rendered in light grey.</span>
</div>
"""

STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 24px; color: #111; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  h2 { font-size: 14px; margin-top: 24px; margin-bottom: 8px; color: #555; }
  .meta { color: #555; font-size: 12px; margin-bottom: 16px; }
  .prompt { background: #f6f6f6; padding: 12px 14px; border-left: 3px solid #888;
            font-family: ui-monospace, Menlo, Monaco, monospace; font-size: 12px;
            white-space: pre-wrap; max-height: 220px; overflow-y: auto; }
  .completion { font-family: ui-monospace, Menlo, Monaco, monospace; font-size: 13px;
                line-height: 1.65; padding: 14px; border: 1px solid #ddd;
                border-radius: 4px; background: #fff; word-wrap: break-word; }
  .tok { padding: 1px 2px; border-radius: 2px; margin: 0 1px; color: #111; }
  .tok.ignored { background: #e5e5e5; color: #555; }
  .nl, .ws { color: #aaa; }
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


def render_html(
    *,
    sample_id: str,
    score_source: str,
    prompt_text: str,
    completion_tokens: List[str],
    completion_scores: List[Optional[float]],
) -> str:
    spans = []
    for tok, score in zip(completion_tokens, completion_scores):
        tok_html = _escape_token(tok)
        if score is None:
            spans.append(f"<span class='tok ignored' title='no score'>{tok_html}</span>")
        else:
            r, g, b = _score_to_rgb(score)
            tooltip = f"{score:.3f}"
            spans.append(
                f"<span class='tok' style='background:rgb({r},{g},{b})' "
                f"title='{tooltip}'>{tok_html}</span>"
            )

    body = (
        f"<h1>{html.escape(sample_id)} — {html.escape(score_source)}</h1>"
        f"<div class='meta'>Color scale fixed to [0, 1]. Hover a token to see the score.</div>"
        f"<h2>Prompt</h2><div class='prompt'>{html.escape(prompt_text)}</div>"
        f"<h2>Completion (per-token {html.escape(score_source)})</h2>"
        f"<div class='completion'>{''.join(spans)}</div>"
        f"{LEGEND_HTML}"
    )
    return f"<!doctype html><html><head><meta charset='utf-8'>{STYLE}</head><body>{body}</body></html>"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, id_col = load_target_rows(args.hf_repo, args.split, args.subset, args.sample_ids)
    sample_ids_in_order = list(rows[id_col])

    model, probe, tokenizer = load_probe(args.probe_dir, args.model_name, args.model_dtype)

    viz_ds = build_dataset_for_rows(
        rows,
        tokenizer,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
    )

    for sample_idx, sample_id in enumerate(sample_ids_in_order):
        print(f"\n[viz] === Sample {sample_id} ===")
        item = viz_ds[sample_idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["classification_labels"]
        lm_labels = item["lm_labels"]

        completion_mask = (lm_labels != -100) & (attention_mask == 1)
        completion_idx = torch.nonzero(completion_mask, as_tuple=False).flatten().tolist()
        if not completion_idx:
            print(f"[viz] No completion tokens for {sample_id}, skipping")
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

        det_scores_full = labels.cpu().numpy()
        det_scores = [
            (None if det_scores_full[i] == -100.0 else float(det_scores_full[i]))
            for i in completion_idx
        ]

        probe_scores_full = score_sample(probe, item)
        probe_scores = [float(probe_scores_full[i]) for i in completion_idx]

        det_path = args.output_dir / f"{sample_id}_deterministic.html"
        det_path.write_text(render_html(
            sample_id=sample_id,
            score_source="deterministic checker score",
            prompt_text=prompt_text,
            completion_tokens=completion_tokens,
            completion_scores=det_scores,
        ))
        print(f"[viz] wrote {det_path}")

        probe_path = args.output_dir / f"{sample_id}_probe.html"
        probe_path.write_text(render_html(
            sample_id=sample_id,
            score_source="probe sigmoid score",
            prompt_text=prompt_text,
            completion_tokens=completion_tokens,
            completion_scores=probe_scores,
        ))
        print(f"[viz] wrote {probe_path}")

        # Also dump raw arrays so the comparison can be re-rendered later.
        np.savez(
            args.output_dir / f"{sample_id}_scores.npz",
            completion_token_indices=np.array(completion_idx, dtype=np.int64),
            deterministic=np.array([s if s is not None else np.nan for s in det_scores]),
            probe=np.array(probe_scores),
            completion_tokens=np.array(completion_tokens, dtype=object),
        )

    summary = {
        "probe_dir": str(args.probe_dir),
        "hf_repo": args.hf_repo,
        "split": args.split,
        "subset": args.subset,
        "id_column": id_col,
        "sample_ids": sample_ids_in_order,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[viz] done — files in {args.output_dir}")


if __name__ == "__main__":
    main()
