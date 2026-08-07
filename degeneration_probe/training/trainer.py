"""Transformers trainer for the single degeneration task."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import Trainer

from degeneration_probe.config import TrainingConfig
from degeneration_probe.evaluation.evaluate import evaluate_probe
from degeneration_probe.training.loss import compute_degeneration_loss


class ProbeTrainer(Trainer):
    def __init__(
        self,
        *,
        cfg: TrainingConfig,
        validation_dataset,
        final_evaluation_datasets: Dict[str, object],
        pos_weight: Optional[float],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.loss_name = cfg.loss.name
        self.validation_dataset = validation_dataset
        self.final_evaluation_datasets = final_evaluation_datasets
        self.pos_weight = pos_weight
        self._last_diagnostic_step = -1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = model.module.device if isinstance(model, nn.DataParallel) else model.device
        targets = inputs["targets"].to(device)
        target_mask = inputs["target_mask"].to(device)
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
        logits = outputs["probe_logits"]
        loss, active_tokens = compute_degeneration_loss(
            self.loss_name,
            logits,
            targets,
            target_mask,
            pos_weight=self.pos_weight,
        )
        if self.state.global_step != self._last_diagnostic_step:
            valid = target_mask & torch.isfinite(targets)
            predictions = torch.sigmoid(logits[valid].float())
            diagnostics = {
                f"train/{self.loss_name}_loss": float(loss.detach().item()),
                "train/active_tokens": active_tokens,
                "train/target_mean": float(targets[valid].float().mean().item()),
                "train/prediction_mean": float(predictions.mean().item()),
            }
            if self.loss_name == "bce" and self.pos_weight is not None:
                diagnostics["train/pos_weight"] = float(self.pos_weight)
            self.log(diagnostics)
            self._last_diagnostic_step = self.state.global_step
        if return_outputs:
            outputs["loss"] = loss
            return loss, outputs
        return loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        probe_parameters = []
        lora_parameters = []
        unexpected = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("linear.") or name.startswith("pre_head_norm."):
                probe_parameters.append(parameter)
            elif "lora" in name.lower():
                lora_parameters.append(parameter)
            else:
                unexpected.append(name)
        if unexpected:
            raise ValueError(f"Unexpected trainable parameters: {unexpected}")
        if not probe_parameters:
            raise ValueError("Scalar linear probe parameters were not found")
        groups = [
            {
                "params": probe_parameters,
                "lr": self.cfg.optimizer.probe_learning_rate,
                "weight_decay": self.cfg.optimizer.weight_decay,
            }
        ]
        if lora_parameters:
            groups.append(
                {
                    "params": lora_parameters,
                    "lr": self.cfg.optimizer.lora_learning_rate,
                    "weight_decay": self.cfg.optimizer.weight_decay,
                }
            )
        self.optimizer = AdamW(groups)
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        dataset = eval_dataset if eval_dataset is not None else self.validation_dataset
        dataloader = self.get_eval_dataloader(dataset)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader).eval()
        metrics = evaluate_probe(
            model,
            dataloader,
            loss_name=self.loss_name,
            prefix="val",
            pos_weight=self.pos_weight,
            metric_names=self.cfg.validation.metrics,
        )
        metrics["eval_loss"] = metrics["val/loss"]
        self.log(metrics)
        return metrics

    def evaluate_final(self) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        for split, dataset in self.final_evaluation_datasets.items():
            dataloader = self.get_eval_dataloader(dataset)
            model = self._wrap_model(self.model, training=False, dataloader=dataloader).eval()
            metrics = evaluate_probe(
                model,
                dataloader,
                loss_name=self.loss_name,
                prefix=split,
                pos_weight=self.pos_weight,
                metric_names=self.cfg.validation.metrics,
            )
            self.log(metrics)
            all_metrics.update(metrics)
        return all_metrics
