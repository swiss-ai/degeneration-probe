"""Transformers trainer for the single degeneration task."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from transformers import Trainer

from degeneration_probe.config import TrainingConfig
from degeneration_probe.evaluation.evaluate import evaluate_probe
from degeneration_probe.training.loss import compute_degeneration_loss


class ProbeTrainer(Trainer):
    """Trains the probe head and, when enabled, the adapters underneath it.

    Checkpointing is overridden throughout: the trained parameters are a linear
    head plus optional adapters, a few megabytes in total, while the frozen
    language model they sit on is fully described by the configuration. Saving
    only the trained part keeps checkpoints cheap enough to write on the
    validation cadence, which is what makes a run both resumable and able to
    keep its best step rather than only its last one.
    """

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
        if self.args.n_gpu > 1:
            # The probe reads its layer through a forward hook on the language
            # model. Replicating the module across devices leaves the capture on
            # a different device from the head that consumes it, so the failure
            # is a device mismatch deep in a forward pass rather than anything
            # readable. One visible device keeps that impossible.
            raise ValueError(
                f"{self.args.n_gpu} visible GPUs: the degeneration probe runs on a single "
                "device. Restrict the job with CUDA_VISIBLE_DEVICES."
            )
        self.cfg = cfg
        self.loss_name = cfg.loss.name
        # Built whether or not the run stops itself: the rule also defines the
        # scalar a checkpoint is ranked by, and that is needed either way.
        self.rule = cfg.stopping.as_rule()
        self.validation_dataset = validation_dataset
        self.final_evaluation_datasets = final_evaluation_datasets
        self.pos_weight = pos_weight
        self._diagnostics: Dict[str, float] = {}
        self._diagnostic_batches = 0
        groups = getattr(getattr(self, "model", None), "head_parameters", None)
        self._head_parameter_groups = groups() if callable(groups) else None
        self._per_head_clip = float(self.cfg.runtime.max_grad_norm or 0.0)

    def _clip_head_gradients(self, *_args, **_kwargs) -> None:
        """Clip each head's gradient on its own when a probe has several.

        A norm taken across every head at once would rescale all of them
        whenever any one exceeded the limit, which is the single place a joint
        run stops matching separate runs. The global clip is switched off in the
        arguments for a multi-head probe, and this restores the same limit per
        head, where it means what it meant for one.

        Attached to the optimizer rather than to the training step, because the
        limit applies to the gradient a step is taken on: that is the one
        accumulated across micro-batches, not the one any single micro-batch
        produced.
        """
        if not self._head_parameter_groups or self._per_head_clip <= 0:
            return
        for parameters in self._head_parameter_groups:
            torch.nn.utils.clip_grad_norm_(parameters, self._per_head_clip)

    # ---- training ----

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = model.device
        targets = inputs["targets"].to(device)
        target_mask = inputs["target_mask"].to(device)
        outputs = model(**self._model_inputs(inputs, device))
        logits = outputs["probe_logits"]
        loss, active_tokens = compute_degeneration_loss(
            self.loss_name,
            logits,
            targets,
            target_mask,
            pos_weight=self.pos_weight,
        )
        self._accumulate_diagnostics(logits, targets, target_mask, active_tokens)
        if return_outputs:
            outputs["loss"] = loss
            return loss, outputs
        return loss

    @staticmethod
    def _model_inputs(inputs, device):
        """Whichever features the configured regime produced."""
        if "features" in inputs:
            return {"features": inputs["features"].to(device)}
        return {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }

    @torch.no_grad()
    def _accumulate_diagnostics(self, logits, targets, target_mask, active_tokens: int) -> None:
        """Collect per-batch statistics, averaged into the next log record.

        Emitting these on their own cadence would double the number of records
        and leave the history with gaps in every column; folding them into the
        regular training record keeps one row per logged step.
        """
        valid = target_mask.bool() & torch.isfinite(targets)
        if not valid.any():
            return
        predictions = torch.sigmoid(logits[valid].float())
        batch = {
            "train/active_tokens": float(active_tokens),
            "train/target_mean": float(targets[valid].float().mean()),
            "train/prediction_mean": float(predictions.mean()),
            "train/prediction_std": float(predictions.std(unbiased=False)),
        }
        for key, value in batch.items():
            self._diagnostics[key] = self._diagnostics.get(key, 0.0) + value
        self._diagnostic_batches += 1

    def _drain_diagnostics(self) -> Dict[str, float]:
        if not self._diagnostic_batches:
            return {}
        drained = {
            key: value / self._diagnostic_batches for key, value in self._diagnostics.items()
        }
        if self.pos_weight is not None:
            drained["train/pos_weight"] = float(self.pos_weight)
        self._diagnostics.clear()
        self._diagnostic_batches = 0
        return drained

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if "loss" in logs:
            logs = {**logs, **self._drain_diagnostics()}
        super().log(logs, start_time)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model
        probe_parameters = []
        lora_parameters = []
        unexpected = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            # A multi-head probe nests the same two modules under heads.<i>.
            tail = name.split("heads.", 1)[-1].split(".", 1)[-1] if name.startswith("heads.") else name
            if tail.startswith("linear.") or tail.startswith("pre_head_norm."):
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
        if self._head_parameter_groups:
            self.optimizer.register_step_pre_hook(self._clip_head_gradients)
        return self.optimizer

    def _get_train_sampler(self, *args, **kwargs):
        """Read a composed dataset in order, so its batches survive intact.

        The composition rule already fixed each batch's class mix and domain
        spread by the order it produced; shuffling on top would undo exactly
        that.
        """
        if getattr(self.train_dataset, "compose", False):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    # ---- checkpointing ----

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Write the trained parameters only, never the frozen backbone."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.model.save(Path(output_dir))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        (model or self.model).load_weights(Path(resume_from_checkpoint))

    def _load_best_model(self) -> None:
        if not self.state.best_model_checkpoint:
            return
        self._load_from_checkpoint(self.state.best_model_checkpoint)

    # ---- evaluation ----

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        """Score the monitoring split and expose the result under both names.

        Metrics are namespaced (``val/...``) for readability in the tracker and
        mirrored with an ``eval_`` prefix, which is the form best-checkpoint
        selection and early stopping key off, so any monitored quantity can
        drive selection without further plumbing.
        """
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
            rule=self.rule,
        )
        self.log(dict(metrics))
        for key, value in list(metrics.items()):
            metrics[f"eval_{key.split('/', 1)[1].replace('/', '_')}"] = value

        selection = self.cfg.checkpoint.metric_for_best_model
        wanted = selection if selection.startswith("eval_") else f"eval_{selection}"
        if self.cfg.checkpoint.keep_best and wanted not in metrics:
            raise ValueError(
                f"selection metric {selection!r} was not produced by validation. The rule's "
                "objective needs healthy rollouts to set a threshold against and degenerate "
                "rollouts whose frontier falls inside the scored completion, so check that the "
                f"monitoring split holds both. Available: {sorted(metrics)}"
            )
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def evaluate_final(self) -> Dict[str, float]:
        """Score every evaluation split once, under its own namespace.

        Kept separate from the monitoring namespace so the final full-split
        numbers never overwrite the last point of the training-time curve.
        """
        all_metrics: Dict[str, float] = {}
        for split, dataset in self.final_evaluation_datasets.items():
            dataloader = self.get_eval_dataloader(dataset)
            model = self._wrap_model(self.model, training=False, dataloader=dataloader).eval()
            metrics = evaluate_probe(
                model,
                dataloader,
                loss_name=self.loss_name,
                prefix=f"final/{split}",
                pos_weight=self.pos_weight,
                metric_names=self.cfg.validation.metrics,
                rule=self.rule,
            )
            self.log(metrics)
            all_metrics.update(metrics)
        return all_metrics
