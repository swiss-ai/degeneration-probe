"""Loss-based evaluation for the degeneration probe.

Validation produces the same record the checkpoint rule reads: a threshold set
to a false-alarm budget, coverage of the loop, and coverage of the run-up at
each warning width. That record is the one definition, shared with the replay
of saved checkpoints, so a rule applied while a run trains and the same rule
applied afterwards cannot disagree.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from degeneration_probe.evaluation.metrics import build_validation_metrics
from degeneration_probe.training.loss import compute_degeneration_loss

# Metrics that a depth is better for having more of. Everything else is either
# lower-is-better or a description of the operating point rather than a score.
GREATER_IS_BETTER = {
    "rollout_auc",
    "rollout_ap",
    "prediction_std",
    "recall_at_budget",
    "in_pattern_recall",
    "warning_recall_128",
    "warning_recall_256",
    "selection_score",
}


def monitor_record(
    negative_peaks: Sequence[float],
    positive_scores: Sequence[np.ndarray],
    onsets: Sequence[int],
) -> Dict[str, float]:
    """What one depth reports at one step, on the monitoring split.

    Delegates to the single definition in ``head_selection`` rather than
    restating it, which is what keeps a live run and a replay of its saved
    checkpoints measuring the same quantity.
    """
    from degeneration_probe.evaluation.head_selection import STEERING_BUDGET, validation_record

    if not len(negative_peaks) or not len(positive_scores):
        return {}
    return validation_record(
        negative_peaks=negative_peaks,
        positive_scores=positive_scores,
        onsets=onsets,
        budget=STEERING_BUDGET,
    )


def selection_score(record: Dict[str, float], rule) -> Optional[float]:
    """The scalar a depth is ranked by, or ``None`` when the rule cannot speak.

    Coverage of the run-up, once coverage inside the loop clears the floor. A
    depth below the floor scores below every depth above it rather than being
    dropped, because the trainer's own best-model tracking needs a total order
    and a missing value would silently keep whatever came before.

    The latch that the full rule applies to eligibility is deliberately absent
    here: latching is a statement about a trajectory, and this is a statement
    about one step. The trajectory is where the rule is actually applied.
    """
    if rule is None or not record:
        return None
    objective = record.get(rule.objective)
    inside = record.get("in_pattern_recall")
    if objective is None or inside is None or not math.isfinite(objective):
        return None
    return float(objective) if inside >= rule.floor else -1.0


def _completion_scores(probabilities, valid, row: int) -> np.ndarray:
    """One rollout's scores over the tokens it is measured on, in order."""
    return probabilities[row][valid[row]].detach().cpu().numpy().astype(np.float32)


def _usable_onset(onset, length: int) -> Optional[int]:
    """The frontier, when it lies inside the scored part of the rollout.

    A rollout longer than the configured completion budget is scored up to that
    budget, so a loop starting past it has no run-up to measure and no in-loop
    tokens to cover. Such a rollout contributes to neither, rather than
    contributing a window that runs off the end.
    """
    if onset is None:
        return None
    try:
        onset = int(onset)
    except (TypeError, ValueError):
        return None
    return onset if 0 <= onset < length else None


@torch.no_grad()
def evaluate_probe(
    probe,
    dataloader,
    *,
    loss_name: str,
    prefix: str,
    pos_weight: Optional[float] = None,
    metric_names: Iterable[str] = (),
    rule=None,
) -> Dict[str, float]:
    """Compute token-weighted loss, collapse diagnostics and the rule's record.

    A probe with a head per depth is scored in the same single pass over the
    data, since re-reading the split once per head would cost the whole saving
    that training them together bought. Each head's metrics are namespaced by
    its layer, and the head the rule ranks highest is repeated unprefixed so
    that selection and early stopping have a scalar to key off.
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
            rule=rule,
        )
    device = getattr(probe, "device", next(probe.parameters()).device)
    optional_metrics = build_validation_metrics(metric_names)
    # Accumulated per rollout so a rollout-level metric can be formed: the
    # score a rollout is judged by is the highest it ever reaches, which is
    # exactly the quantity a decision threshold acts on. Degenerate rollouts
    # keep every token as well, because coverage is a statement about tokens
    # and the threshold is not known until the healthy ones have been read.
    rollout_peak: Dict[tuple, float] = {}
    rollout_label: Dict[tuple, bool] = {}
    positive_scores: Dict[tuple, np.ndarray] = {}
    positive_onsets: Dict[tuple, int] = {}
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
        labels_present = batch.get("is_positive", [False] * len(batch["prompt_id"]))
        onsets_present = batch.get("onset_position", [None] * len(batch["prompt_id"]))
        for row, key in enumerate(zip(batch["prompt_id"], batch["rollout_idx"])):
            row_valid = valid[row]
            if not bool(row_valid.any()):
                continue
            is_positive = bool(labels_present[row])
            scores = _completion_scores(probabilities, valid, row)
            rollout_peak[key] = max(rollout_peak.get(key, 0.0), float(scores.max()))
            rollout_label[key] = is_positive
            if is_positive:
                onset = _usable_onset(onsets_present[row], scores.size)
                if onset is not None:
                    positive_scores[key] = scores
                    positive_onsets[key] = onset
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
    record = monitor_record(
        [peak for key, peak in rollout_peak.items() if not rollout_label[key]],
        [positive_scores[key] for key in positive_scores],
        [positive_onsets[key] for key in positive_scores],
    )
    for key, value in record.items():
        metrics[f"{prefix}/{key}"] = value
    score = selection_score(record, rule)
    if score is not None:
        metrics[f"{prefix}/selection_score"] = score
    metrics[f"{prefix}/rollouts"] = len(rollout_peak)
    metrics[f"{prefix}/positive_rollouts"] = int(sum(labels))
    metrics[f"{prefix}/measured_positive_rollouts"] = len(positive_scores)
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
    rule=None,
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
    # One array per degenerate rollout, shaped [depths, tokens]: the run-up is
    # measured per depth and the threshold it is measured against is not known
    # until the healthy rollouts have been read.
    positive_scores: Dict[tuple, np.ndarray] = {}
    positive_onsets: Dict[tuple, int] = {}
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
        onsets_present = batch.get("onset_position", [None] * len(batch["prompt_id"]))
        for row, key in enumerate(zip(batch["prompt_id"], batch["rollout_idx"])):
            if not bool(rows_present[row]):
                continue
            labels[key] = bool(is_positive[row])
            row_values = row_peaks[row]
            for index in range(heads):
                value = float(row_values[index])
                if value > peaks[index].get(key, -1.0):
                    peaks[index][key] = value
            if not labels[key] or key in positive_scores:
                continue
            # [tokens, depths] to [depths, tokens], on the host: a monitoring
            # split holds a hundred or so degenerate rollouts, so this is tens
            # of megabytes rather than anything that needs managing.
            scores = probabilities[row][valid[row]].t().contiguous().cpu().numpy()
            onset = _usable_onset(onsets_present[row], scores.shape[1])
            if onset is not None:
                positive_scores[key] = scores.astype(np.float32)
                positive_onsets[key] = onset

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
        f"{prefix}/measured_positive_rollouts": len(positive_scores),
    }
    if loss_name == "bce" and pos_weight is not None:
        metrics[f"{prefix}/pos_weight"] = float(pos_weight)

    ordered_positives = list(positive_scores)
    onsets = [positive_onsets[key] for key in ordered_positives]
    per_head: List[Dict[str, float]] = []
    for index, layer in enumerate(layer_indices):
        name = f"{prefix}/layer{layer:02d}"
        mean = float(prediction_sum[index]) / total_tokens
        head_metrics = {
            "loss": float(loss_total[index]) / total_tokens,
            "loss_unweighted": float(plain_total[index]) / total_tokens,
            "prediction_mean": mean,
            "prediction_std": math.sqrt(
                max(0.0, float(prediction_sq_sum[index]) / total_tokens - mean**2)
            ),
            "prediction_positive_rate": float(predicted_positive[index]) / total_tokens,
        }
        head_labels = [labels[key] for key in peaks[index]]
        if any(head_labels) and not all(head_labels):
            from sklearn.metrics import average_precision_score, roc_auc_score

            values = [peaks[index][key] for key in peaks[index]]
            head_metrics["rollout_auc"] = float(roc_auc_score(head_labels, values))
            head_metrics["rollout_ap"] = float(
                average_precision_score(head_labels, values)
            )
        record = monitor_record(
            [peak for key, peak in peaks[index].items() if not labels[key]],
            [positive_scores[key][index] for key in ordered_positives],
            onsets,
        )
        head_metrics.update(record)
        score = selection_score(record, rule)
        if score is not None:
            head_metrics["selection_score"] = score
        per_head.append(head_metrics)
        for key, value in head_metrics.items():
            metrics[f"{name}/{key}"] = value

    metrics.update(_headline(per_head, prefix, rule))
    return metrics


def _headline(per_head: List[Dict[str, float]], prefix: str, rule) -> Dict[str, float]:
    """The run's unprefixed numbers: one depth's row, not a best-of per column.

    A run's headline is the probe it would actually be used at, so every number
    in it has to come from the same head. Taking each column's best separately
    would report a loss from one depth beside a coverage from another and call
    the result a run, which is a scorer that does not exist.

    Depths are ordered by the rule's score and, below its floor where every
    depth scores alike, by how much of the loop each has learned to cover, so
    that early in a run the headline is the leading depth rather than the first
    one. Without a rule there is no such thing as the run's depth, and each
    column's best is reported instead.
    """
    if not per_head:
        return {}
    ranked = [head.get("selection_score") for head in per_head]
    if rule is not None and any(value is not None for value in ranked):
        best = max(
            range(len(per_head)),
            key=lambda index: (
                ranked[index] is not None,
                ranked[index] if ranked[index] is not None else -np.inf,
                per_head[index].get("in_pattern_recall", -np.inf),
            ),
        )
        headline = dict(per_head[best])
    else:
        headline = {}
        for head in per_head:
            for key, value in head.items():
                better = key in GREATER_IS_BETTER
                current = headline.get(key)
                if current is None or (value > current if better else value < current):
                    headline[key] = value
    # The collapse guard asks whether the probe has stopped distinguishing
    # anything, and the answer is no while any head still spreads its scores. A
    # guard reading one depth's spread would kill a run for the state of a depth
    # nobody would use, which early on is whichever depth the ranking has not
    # separated yet.
    spreads = [head["prediction_std"] for head in per_head if "prediction_std" in head]
    if spreads:
        headline["prediction_std"] = max(spreads)
    return {f"{prefix}/{key}": value for key, value in headline.items()}
