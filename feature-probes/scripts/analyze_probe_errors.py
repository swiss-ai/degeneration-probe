"""Error analysis for a trained probe checkpoint.

Loads a saved probe, runs it on the validation set, and answers:
- Is the label distribution actually balanced or skewed towards "non-degenerated"?
- Is the probe collapsing predictions to a narrow band (e.g. ~0.5)?
- Where exactly does it fail (false negatives on degenerations, calibration)?
- What are the worst-mispredicted tokens (with decoded context)?

Example
-------
    python scripts/analyze_probe_errors.py \\
        --probe-id apertus_apertus_stability_repetition_balanced_layer30 \\
        --model apertus \\
        --layer 30 \\
        --probe-dtype float32 \\
        --norm layernorm \\
        --lora-layers all \\
        --epoch 1 \\
        --batch-size 4 \\
        --top-k 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_probes.config import ProbeConfig
from feature_probes.data.dataset import (
    TokenizedProbingDatasetConfig,
    create_probing_dataset,
    tokenized_probing_collate_fn,
)
from feature_probes.probes.value_head_probe import setup_probe
from feature_probes.utils.model_utils import (
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from transformers import PreTrainedModel


MODEL_NAMES = {
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

DEFAULT_EVAL_DS = {
    "dataset_id": "repetition_validation",
    "hf_repo": "lorenzo0312/degeneration-probe-instruct-token-level-balanced",
    "subset": None,
    "split": "validation",
    "max_length": 4608,
    "max_completion_length": 4096,
    "prompt_truncation_side": "left",
    "pos_weight": 1.0,
    "neg_weight": 1.0,
    "default_ignore": True,
    "shuffle": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze where a trained probe makes mistakes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe-id", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--probe-dtype", required=True, choices=["float32", "bfloat16", "float16", "auto"])
    p.add_argument("--norm", required=True, choices=["none", "layernorm", "rmsnorm", "l2"])
    p.add_argument("--lora-layers", required=True, choices=["none", "all"])
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--epoch", type=int, default=None,
                   help="Load epoch_{N} checkpoint instead of probe_path root.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--degeneration-threshold", type=float, default=0.8,
                   help="Score above which a token is considered degenerated.")
    p.add_argument("--pred-threshold", type=float, default=0.5,
                   help="Score above which the probe is considered to predict degeneration.")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-K worst false-negative/false-positive tokens to dump.")
    p.add_argument("--context-window", type=int, default=20,
                   help="Tokens before/after the error to decode for context.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write report.json and worst_errors.csv "
                        "(default: <probe_path>/error_analysis/).")
    p.add_argument("--dataset-id", default=DEFAULT_EVAL_DS["dataset_id"])
    p.add_argument("--dataset-hf-repo", default=DEFAULT_EVAL_DS["hf_repo"])
    p.add_argument("--dataset-split", default=DEFAULT_EVAL_DS["split"])
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Save full per-token predictions to predictions.npz.")
    return p.parse_args()


def _bucketize(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Right-open buckets [e0,e1), [e1,e2), …, [e_{n-1}, e_n]."""
    idx = np.digitize(values, edges[1:-1], right=False)
    return idx  # 0..len(edges)-2


def _bucket_label(edges: list[float], i: int) -> str:
    return f"[{edges[i]:.2f}, {edges[i + 1]:.2f}{')' if i < len(edges) - 2 else ']'}"


def main():
    args = parse_args()

    probe_config = ProbeConfig(
        probe_id=args.probe_id,
        model_name=MODEL_NAMES[args.model],
        layer=args.layer,
        probe_dtype=args.probe_dtype,
        normalize_before_head=args.norm,
        lora_layers=args.lora_layers,
        load_from="disk",
    )

    if args.epoch is not None:
        probe_config.probe_path = probe_config.probe_path / f"epoch_{args.epoch}"
    if not probe_config.probe_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {probe_config.probe_path}. "
            "Has at least one epoch finished?"
        )
    print(f"[analyze] Loading probe from: {probe_config.probe_path}")

    print(f"[analyze] Loading model: {probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        probe_config.model_name,
        torch_dtype=resolve_torch_dtype(args.model_dtype),
    )

    model, probe = setup_probe(cast(PreTrainedModel, model), probe_config, seed=args.seed)
    probe.eval()

    ds_cfg = TokenizedProbingDatasetConfig(
        dataset_id=args.dataset_id,
        hf_repo=args.dataset_hf_repo,
        subset=DEFAULT_EVAL_DS["subset"],
        split=args.dataset_split,
        max_length=DEFAULT_EVAL_DS["max_length"],
        max_completion_length=DEFAULT_EVAL_DS["max_completion_length"],
        prompt_truncation_side=DEFAULT_EVAL_DS["prompt_truncation_side"],
        pos_weight=DEFAULT_EVAL_DS["pos_weight"],
        neg_weight=DEFAULT_EVAL_DS["neg_weight"],
        default_ignore=DEFAULT_EVAL_DS["default_ignore"],
        shuffle=False,
    )
    print(f"[analyze] Loading dataset {args.dataset_hf_repo} / {args.dataset_split}")
    dataset = create_probing_dataset(ds_cfg, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=tokenized_probing_collate_fn,
        shuffle=False,
    )
    print(f"[analyze] {len(dataset)} samples in eval set")

    preds: list[float] = []
    targets: list[float] = []
    sample_ids: list[int] = []
    token_pos: list[int] = []
    input_ids_keep: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(probe.device)
            attention_mask = batch["attention_mask"].to(probe.device)
            classification_labels = batch["classification_labels"].to(probe.device)
            lm_labels = batch["lm_labels"].to(probe.device)

            outputs = probe(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=lm_labels,
            )
            probe_logits = outputs["probe_logits"].squeeze(-1)
            probe_scores = torch.sigmoid(probe_logits).float()

            valid_mask = (attention_mask == 1) & (classification_labels != -100.0)
            for b in range(input_ids.size(0)):
                m = valid_mask[b]
                positions = torch.nonzero(m, as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    continue
                p = probe_scores[b, positions].cpu().numpy()
                t = classification_labels[b, positions].cpu().numpy()
                keep_idx = len(input_ids_keep)
                preds.extend(p.tolist())
                targets.extend(t.tolist())
                sample_ids.extend([keep_idx] * positions.numel())
                token_pos.extend(positions.cpu().tolist())
                input_ids_keep.append(input_ids[b].cpu())

            if (batch_i + 1) % 10 == 0:
                print(f"[analyze] batch {batch_i + 1}/{len(dataloader)}")

    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
    token_pos_np = np.asarray(token_pos, dtype=np.int64)

    n_total = preds.size
    print(f"\n[analyze] Total valid tokens scored: {n_total}")

    # ───── Distributions ──────────────────────────────────────────────────────
    edges = [0.0, 0.2, 0.5, 0.8, 1.0001]
    n_buckets = len(edges) - 1

    target_buckets = _bucketize(targets, edges)
    pred_buckets = _bucketize(preds, edges)

    target_counts = np.bincount(target_buckets, minlength=n_buckets)
    pred_counts = np.bincount(pred_buckets, minlength=n_buckets)

    # ───── Per-target-bucket conditional MSE ──────────────────────────────────
    per_bucket_mse = []
    per_bucket_mean_pred = []
    for i in range(n_buckets):
        mask = target_buckets == i
        if mask.sum() == 0:
            per_bucket_mse.append(None)
            per_bucket_mean_pred.append(None)
            continue
        per_bucket_mse.append(float(np.mean((preds[mask] - targets[mask]) ** 2)))
        per_bucket_mean_pred.append(float(np.mean(preds[mask])))

    # ───── Calibration (per-pred-bucket mean target) ──────────────────────────
    per_pred_bucket_mean_target = []
    for i in range(n_buckets):
        mask = pred_buckets == i
        if mask.sum() == 0:
            per_pred_bucket_mean_target.append(None)
            continue
        per_pred_bucket_mean_target.append(float(np.mean(targets[mask])))

    # ───── Binary degeneration metrics at degeneration_threshold ──────────────
    deg_th = args.degeneration_threshold
    pred_th = args.pred_threshold
    y_true = (targets >= deg_th).astype(np.int64)
    y_pred = (preds >= pred_th).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # ───── "Always-predict-low" diagnostic ────────────────────────────────────
    miss_rate_on_degen = 1.0 - recall
    pred_std = float(preds.std())
    pred_p05, pred_p95 = float(np.percentile(preds, 5)), float(np.percentile(preds, 95))

    # ───── Overall MSE ────────────────────────────────────────────────────────
    overall_mse = float(np.mean((preds - targets) ** 2))
    overall_target_mean = float(targets.mean())
    overall_target_std = float(targets.std())
    overall_pred_mean = float(preds.mean())

    # ───── Worst errors ───────────────────────────────────────────────────────
    abs_err = np.abs(preds - targets)
    # Worst FN: target >= deg_th and predicted < pred_th, sorted by abs error
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    def topk(mask: np.ndarray, k: int) -> np.ndarray:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return idx
        order = np.argsort(-abs_err[idx])[:k]
        return idx[order]

    worst_fn = topk(fn_mask, args.top_k)
    worst_fp = topk(fp_mask, args.top_k)

    def decode_context(global_token_idx: int) -> dict:
        sid = int(sample_ids_np[global_token_idx])
        tpos = int(token_pos_np[global_token_idx])
        ids = input_ids_keep[sid]
        lo = max(0, tpos - args.context_window)
        hi = min(ids.size(0), tpos + args.context_window + 1)
        before = tokenizer.decode(ids[lo:tpos], skip_special_tokens=False)
        token = tokenizer.decode(ids[tpos:tpos + 1], skip_special_tokens=False)
        after = tokenizer.decode(ids[tpos + 1:hi], skip_special_tokens=False)
        return {
            "sample_id": sid,
            "token_pos": tpos,
            "target": float(targets[global_token_idx]),
            "pred": float(preds[global_token_idx]),
            "abs_err": float(abs_err[global_token_idx]),
            "context_before": before,
            "token": token,
            "context_after": after,
        }

    worst_fn_records = [decode_context(int(i)) for i in worst_fn]
    worst_fp_records = [decode_context(int(i)) for i in worst_fp]

    # ───── Output ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ERROR ANALYSIS REPORT")
    print("=" * 72)
    print(f"probe_id            : {args.probe_id}")
    print(f"checkpoint          : {probe_config.probe_path}")
    print(f"dataset             : {args.dataset_hf_repo} / {args.dataset_split}")
    print(f"valid tokens scored : {n_total}")
    print()
    print(f"target  mean={overall_target_mean:.4f}  std={overall_target_std:.4f}")
    print(f"pred    mean={overall_pred_mean:.4f}  std={pred_std:.4f}  "
          f"p05={pred_p05:.4f}  p95={pred_p95:.4f}")
    print(f"overall MSE         : {overall_mse:.6f}")
    print()
    print("Label distribution (target buckets):")
    for i in range(n_buckets):
        frac = target_counts[i] / max(n_total, 1)
        print(f"  {_bucket_label(edges, i):<14}  n={int(target_counts[i]):>10}  "
              f"({frac:6.2%})")
    print("\nPrediction distribution (pred buckets):")
    for i in range(n_buckets):
        frac = pred_counts[i] / max(n_total, 1)
        print(f"  {_bucket_label(edges, i):<14}  n={int(pred_counts[i]):>10}  "
              f"({frac:6.2%})")

    print("\nConditional MSE & mean pred per TARGET bucket:")
    print(f"  {'target bucket':<14}  {'n':>10}  {'MSE':>10}  {'mean_pred':>10}")
    for i in range(n_buckets):
        n_i = int(target_counts[i])
        mse_i = per_bucket_mse[i]
        mp_i = per_bucket_mean_pred[i]
        print(f"  {_bucket_label(edges, i):<14}  {n_i:>10}  "
              f"{(f'{mse_i:.6f}' if mse_i is not None else '   n/a   '):>10}  "
              f"{(f'{mp_i:.4f}' if mp_i is not None else '  n/a  '):>10}")

    print("\nCalibration (mean target per PRED bucket):")
    print(f"  {'pred bucket':<14}  {'n':>10}  {'mean_target':>12}")
    for i in range(n_buckets):
        n_i = int(pred_counts[i])
        mt = per_pred_bucket_mean_target[i]
        print(f"  {_bucket_label(edges, i):<14}  {n_i:>10}  "
              f"{(f'{mt:.4f}' if mt is not None else '  n/a  '):>12}")

    print(f"\nBinary degeneration (target >= {deg_th}, pred >= {pred_th}):")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}")
    print(f"  miss-rate on actual degenerations: {miss_rate_on_degen:.4f} "
          f"(fraction of high-score tokens predicted as non-degenerated)")

    print("\n--- Diagnostics ---")
    if pred_std < 0.05:
        print(f"  WARNING: prediction std={pred_std:.4f} < 0.05 — probe is "
              f"collapsing predictions to a narrow band.")
    if overall_pred_mean < 0.2 and overall_target_mean > 0.3:
        print(f"  WARNING: probe predicts low ({overall_pred_mean:.3f}) but the "
              f"target distribution has substantial mass above 0.5 — likely "
              f"underfitting to the rare-degeneration class.")
    if recall < 0.2 and (y_true.sum() / max(n_total, 1)) < 0.1:
        print(f"  WARNING: low recall ({recall:.3f}) AND degenerations are rare "
              f"({y_true.sum() / max(n_total, 1):.2%} of tokens). Loss can stay "
              f"low while the probe ignores the positive class.")
    deg_frac = float(y_true.sum() / max(n_total, 1))
    print(f"  Fraction of tokens with target >= {deg_th}: {deg_frac:.2%}")
    print(f"  Trivial-baseline MSE (predict mean target everywhere): "
          f"{float(np.mean((overall_target_mean - targets) ** 2)):.6f}")
    print(f"  Trivial-baseline MSE (predict 0.5 everywhere): "
          f"{float(np.mean((0.5 - targets) ** 2)):.6f}")

    # ───── Write artifacts ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else (
        probe_config.probe_path / "error_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "probe_id": args.probe_id,
        "checkpoint": str(probe_config.probe_path),
        "dataset": {
            "hf_repo": args.dataset_hf_repo,
            "split": args.dataset_split,
            "n_tokens": int(n_total),
        },
        "thresholds": {
            "degeneration_threshold": deg_th,
            "pred_threshold": pred_th,
        },
        "overall": {
            "target_mean": overall_target_mean,
            "target_std": overall_target_std,
            "pred_mean": overall_pred_mean,
            "pred_std": pred_std,
            "pred_p05": pred_p05,
            "pred_p95": pred_p95,
            "mse": overall_mse,
            "trivial_mse_predict_mean": float(
                np.mean((overall_target_mean - targets) ** 2)
            ),
            "trivial_mse_predict_half": float(np.mean((0.5 - targets) ** 2)),
        },
        "buckets": {
            "edges": edges,
            "target_counts": target_counts.tolist(),
            "pred_counts": pred_counts.tolist(),
            "per_target_bucket_mse": per_bucket_mse,
            "per_target_bucket_mean_pred": per_bucket_mean_pred,
            "per_pred_bucket_mean_target": per_pred_bucket_mean_target,
        },
        "binary_metrics": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "miss_rate_on_degen": miss_rate_on_degen,
            "deg_fraction": deg_frac,
        },
        "worst_false_negatives": worst_fn_records,
        "worst_false_positives": worst_fp_records,
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[analyze] Wrote report:    {report_path}")

    if args.save_predictions:
        preds_path = out_dir / "predictions.npz"
        np.savez_compressed(
            preds_path,
            preds=preds.astype(np.float32),
            targets=targets.astype(np.float32),
            sample_ids=sample_ids_np.astype(np.int32),
            token_pos=token_pos_np.astype(np.int32),
            abs_err=abs_err.astype(np.float32),
            meta=np.array(
                [args.probe_id, args.dataset_hf_repo, args.dataset_split, str(deg_th), str(pred_th)],
                dtype=object,
            ),
        )
        print(f"[analyze] Wrote predictions: {preds_path}  ({n_total} tokens)")

    # Also a flat CSV with the worst FN/FP for quick scanning
    import csv
    csv_path = out_dir / "worst_errors.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "sample_id", "token_pos", "target", "pred",
                    "abs_err", "context_before", "token", "context_after"])
        for r in worst_fn_records:
            w.writerow(["FN", r["sample_id"], r["token_pos"], r["target"], r["pred"],
                        r["abs_err"], r["context_before"], r["token"], r["context_after"]])
        for r in worst_fp_records:
            w.writerow(["FP", r["sample_id"], r["token_pos"], r["target"], r["pred"],
                        r["abs_err"], r["context_before"], r["token"], r["context_after"]])
    print(f"[analyze] Wrote worst errors: {csv_path}")


if __name__ == "__main__":
    main()
