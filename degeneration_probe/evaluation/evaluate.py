"""Loss-based evaluation for the degeneration probe."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from degeneration_probe.evaluation.metrics import build_validation_metrics
from degeneration_probe.training.loss import compute_degeneration_loss

# The budget an in-training operating point is reported at. One number rather
# than the reported family, because this exists to steer a run rather than to
# describe it.
MONITOR_BUDGET = 0.01


def operating_point(peaks: Dict[tuple, float], labels: Dict[tuple, bool]) -> Dict[str, float]:
    """Recall when the threshold is set to spend exactly the monitor's budget.

    A rank metric asks whether the ordering is right and saturates long before a
    probe is good, because separating a mostly-degenerate rollout from a healthy
    one is easy. What a deployment turns on is narrower: with false alarms capped
    at a small rate, how many degenerations are actually caught. That question
    stays sensitive where ranking does not, so it is the one worth steering on.

    The threshold comes from the same rule the reported protocol uses, so this
    number and the reported one mean the same thing at a persistence of one.
    """
    from degeneration_probe.evaluation.protocol import threshold_for_budget

    negatives = [peak for key, peak in peaks.items() if not labels.get(key)]
    positives = [peak for key, peak in peaks.items() if labels.get(key)]
    if not negatives or not positives:
        return {}
    tau, realized = threshold_for_budget(np.asarray(negatives), MONITOR_BUDGET)
    caught = sum(1 for peak in positives if peak >= tau)
    return {
        "recall_at_budget": caught / len(positives),
        "budget_tau": tau,
        "budget_realized_fpr": realized,
    }


@torch.no_grad()
def evaluate_probe(
    probe,
    dataloader,
    *,
    loss_name: str,
    prefix: str,
    pos_weight: Optional[float] = None,
    metric_names: Iterable[str] = (),
) -> Dict[str, float]:
    """Compute token-weighted loss and basic collapse diagnostics.

    A probe with a head per depth is scored in the same single pass over the
    data, since re-reading the split once per head would cost the whole saving
    that training them together bought. Each head's metrics are namespaced by
    its layer, and the best of them is repeated unprefixed so that selection and
    early stopping have a scalar to key off.
    """
    probe.eval()
    layer_indices = getattr(probe, "layer_indices", None)
    if layer_indices is not None:
        return _evaluate_multi_head(
            probe,
            dataloader,
            loss_name=loss_name,
            prefix=prefix,
            pos_weight=pos_weight,
            layer_indices=layer_indices,
        )
    device = getattr(probe, "device", next(probe.parameters()).device)
    optional_metrics = build_validation_metrics(metric_names)
    # Accumulated per rollout so a rollout-level metric can be formed: the
    # score a rollout is judged by is the highest it ever reaches, which is
    # exactly the quantity a decision threshold acts on.
    rollout_peak: Dict[tuple, float] = {}
    rollout_label: Dict[tuple, bool] = {}
    total_loss = 0.0
    total_plain_loss = 0.0
    total_tokens = 0
    # The weighted loss is the training objective, but its weight comes from the
    # training stream, whose class balance every selection rule changes. Two
    # recipes therefore score the same monitoring split on two different scales.
    # An unweighted loss alongside it is measured the same way for every recipe,
    # so it can be read across them.
    plain_matches_weighted = loss_name != "bce" or pos_weight is None or float(pos_weight) == 1.0
    target_sum = target_sq_sum = 0.0
    prediction_sum = prediction_sq_sum = 0.0
    predicted_positive = 0

    for batch in dataloader:
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)
        model_inputs = (
            {"features": batch["features"].to(device)}
            if "features" in batch
            else {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
        )
        logits = probe(**model_inputs)["probe_logits"]
        loss, active_tokens = compute_degeneration_loss(
            loss_name,
            logits,
            targets,
            target_mask,
            pos_weight=pos_weight,
        )
        if plain_matches_weighted:
            plain_loss = loss
        else:
            plain_loss, _ = compute_degeneration_loss(
                loss_name, logits, targets, target_mask, pos_weight=None
            )
        valid = target_mask & torch.isfinite(targets)
        valid_targets = targets[valid].float()
        valid_predictions = torch.sigmoid(logits[valid].float())
        probabilities = torch.sigmoid(logits.float())
        for row, key in enumerate(zip(batch["prompt_id"], batch["rollout_idx"])):
            row_valid = valid[row]
            if not bool(row_valid.any()):
                continue
            peak = float(probabilities[row][row_valid].max())
            rollout_peak[key] = max(rollout_peak.get(key, 0.0), peak)
            rollout_label[key] = bool(batch.get("is_positive", [False] * len(batch["prompt_id"]))[row])
        total_loss += float(loss.item()) * active_tokens
        total_plain_loss += float(plain_loss.item()) * active_tokens
        total_tokens += active_tokens
        target_sum += float(valid_targets.sum().item())
        target_sq_sum += float(valid_targets.square().sum().item())
        prediction_sum += float(valid_predictions.sum().item())
        prediction_sq_sum += float(valid_predictions.square().sum().item())
        predicted_positive += int((valid_predictions >= 0.5).sum().item())
        for metric in optional_metrics.values():
            metric.update(
                logits=logits,
                predictions=valid_predictions,
                targets=valid_targets,
                target_mask=valid,
                metadata={
                    "prompt_id": batch["prompt_id"],
                    "rollout_idx": batch["rollout_idx"],
                    "domain": batch["domain"],
                    "split": batch["split"],
                },
            )

    if total_tokens == 0:
        raise ValueError(f"Evaluation split {prefix!r} contains no valid target tokens")
    target_mean = target_sum / total_tokens
    prediction_mean = prediction_sum / total_tokens
    target_variance = max(0.0, target_sq_sum / total_tokens - target_mean**2)
    prediction_variance = max(0.0, prediction_sq_sum / total_tokens - prediction_mean**2)
    loss = total_loss / total_tokens
    metrics = {
        f"{prefix}/loss": loss,
        f"{prefix}/loss_unweighted": total_plain_loss / total_tokens,
        f"{prefix}/valid_tokens": total_tokens,
        f"{prefix}/target_mean": target_mean,
        f"{prefix}/target_std": math.sqrt(target_variance),
        f"{prefix}/prediction_mean": prediction_mean,
        f"{prefix}/prediction_std": math.sqrt(prediction_variance),
        f"{prefix}/prediction_positive_rate": predicted_positive / total_tokens,
    }
    if loss_name == "bce" and pos_weight is not None:
        metrics[f"{prefix}/pos_weight"] = float(pos_weight)

    labels = [rollout_label[key] for key in rollout_peak]
    if any(labels) and not all(labels):
        from sklearn.metrics import average_precision_score, roc_auc_score

        peaks = [rollout_peak[key] for key in rollout_peak]
        metrics[f"{prefix}/rollout_auc"] = float(roc_auc_score(labels, peaks))
        metrics[f"{prefix}/rollout_ap"] = float(average_precision_score(labels, peaks))
    for key, value in operating_point(rollout_peak, rollout_label).items():
        metrics[f"{prefix}/{key}"] = value
    metrics[f"{prefix}/rollouts"] = len(rollout_peak)
    metrics[f"{prefix}/positive_rollouts"] = int(sum(labels))
    for name, metric in optional_metrics.items():
        for key, value in metric.compute().items():
            metrics[f"{prefix}/{name}/{key}"] = float(value)
    return metrics


@torch.no_grad()
def _evaluate_multi_head(
    probe,
    dataloader,
    *,
    loss_name: str,
    prefix: str,
    pos_weight: Optional[float],
    layer_indices,
) -> Dict[str, float]:
    """Score every head of a multi-layer probe in one pass over the split.

    Every quantity is accumulated across the head axis rather than head by
    head. A loop here would multiply the per-batch Python work and the
    device synchronisations by the number of depths, which on a split of
    thousands of rollouts costs hours rather than minutes, and none of it is
    arithmetic the machine needs to do serially.
    """
    device = getattr(probe, "device", next(probe.parameters()).device)
    heads = len(layer_indices)
    peaks = [{} for _ in range(heads)]
    labels: Dict[tuple, bool] = {}
    zeros = torch.zeros(heads, dtype=torch.float64, device=device)
    loss_total = zeros.clone()
    plain_total = zeros.clone()
    prediction_sum = zeros.clone()
    prediction_sq_sum = zeros.clone()
    predicted_positive = zeros.clone()
    total_tokens = 0
    target_sum = target_sq_sum = 0.0
    plain_matches_weighted = (
        loss_name != "bce" or pos_weight is None or float(pos_weight) == 1.0
    )
    weight = (
        torch.tensor(float(pos_weight), device=device)
        if loss_name == "bce" and pos_weight is not None
        else None
    )

    for batch in dataloader:
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)
        logits = probe(features=batch["features"].to(device))["probe_logits"].float()
        valid = target_mask & torch.isfinite(targets)
        active = int(valid.sum().item())
        if not active:
            continue
        total_tokens += active
        valid_targets = targets[valid].float()
        target_sum += float(valid_targets.sum().item())
        target_sq_sum += float(valid_targets.square().sum().item())

        keep = valid.unsqueeze(-1)
        spread = torch.where(valid, targets, torch.zeros_like(targets)).unsqueeze(-1)
        spread = spread.expand_as(logits)
        if loss_name == "bce":
            elementwise = F.binary_cross_entropy_with_logits(
                logits, spread, pos_weight=weight, reduction="none"
            )
            plain_elementwise = (
                elementwise
                if plain_matches_weighted
                else F.binary_cross_entropy_with_logits(
                    logits, spread, reduction="none"
                )
            )
        elif loss_name == "mse":
            elementwise = (torch.sigmoid(logits) - spread).square()
            plain_elementwise = elementwise
        else:
            raise ValueError(f"Unknown degeneration loss {loss_name!r}")
        loss_total += (elementwise * keep).sum(dim=(0, 1)).double()
        plain_total += (plain_elementwise * keep).sum(dim=(0, 1)).double()

        probabilities = torch.sigmoid(logits)
        masked = probabilities * keep
        prediction_sum += masked.sum(dim=(0, 1)).double()
        prediction_sq_sum += (masked * masked).sum(dim=(0, 1)).double()
        predicted_positive += ((probabilities >= 0.5) & keep).sum(dim=(0, 1)).double()

        # The highest score each rollout reaches, per head, in one reduction.
        # Padding is pushed below every real score so it can never be the peak.
        row_peaks = probabilities.masked_fill(~keep, -1.0).amax(dim=1).cpu()
        rows_present = valid.any(dim=1).cpu()
        is_positive = batch.get("is_positive", [False] * len(batch["prompt_id"]))
        for row, key in enumerate(zip(batch["prompt_id"], batch["rollout_idx"])):
            if not bool(rows_present[row]):
                continue
            labels[key] = bool(is_positive[row])
            row_values = row_peaks[row]
            for index in range(heads):
                value = float(row_values[index])
                if value > peaks[index].get(key, -1.0):
                    peaks[index][key] = value

    if total_tokens == 0:
        raise ValueError(f"Evaluation split {prefix!r} contains no valid target tokens")

    target_mean = target_sum / total_tokens
    metrics: Dict[str, float] = {
        f"{prefix}/valid_tokens": total_tokens,
        f"{prefix}/target_mean": target_mean,
        f"{prefix}/target_std": math.sqrt(
            max(0.0, target_sq_sum / total_tokens - target_mean**2)
        ),
        f"{prefix}/rollouts": len(labels),
        f"{prefix}/positive_rollouts": int(sum(labels.values())),
    }
    if loss_name == "bce" and pos_weight is not None:
        metrics[f"{prefix}/pos_weight"] = float(pos_weight)

    best: Dict[str, float] = {}
    for index, layer in enumerate(layer_indices):
        name = f"{prefix}/layer{layer:02d}"
        mean = float(prediction_sum[index]) / total_tokens
        head_metrics = {
            f"{name}/loss": float(loss_total[index]) / total_tokens,
            f"{name}/loss_unweighted": float(plain_total[index]) / total_tokens,
            f"{name}/prediction_mean": mean,
            f"{name}/prediction_std": math.sqrt(
                max(0.0, float(prediction_sq_sum[index]) / total_tokens - mean**2)
            ),
            f"{name}/prediction_positive_rate": float(predicted_positive[index]) / total_tokens,
        }
        ordered = [labels[key] for key in peaks[index]]
        if any(ordered) and not all(ordered):
            from sklearn.metrics import average_precision_score, roc_auc_score

            values = [peaks[index][key] for key in peaks[index]]
            head_metrics[f"{name}/rollout_auc"] = float(roc_auc_score(ordered, values))
            head_metrics[f"{name}/rollout_ap"] = float(
                average_precision_score(ordered, values)
            )
        for key, value in operating_point(peaks[index], labels).items():
            head_metrics[f"{name}/{key}"] = value
        metrics.update(head_metrics)
        for key, value in head_metrics.items():
            suffix = key.rsplit("/", 1)[1]
            # The run's headline number is its best depth, since that is the
            # probe the run would actually be used for. Reported unprefixed so
            # checkpoint selection and the collapse guard keep working unchanged.
            better = suffix in {"rollout_auc", "rollout_ap", "prediction_std", "recall_at_budget"}
            current = best.get(suffix)
            if current is None or (value > current if better else value < current):
                best[suffix] = value
    for suffix, value in best.items():
        metrics[f"{prefix}/{suffix}"] = value
    return metrics
