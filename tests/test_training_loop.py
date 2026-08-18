"""End-to-end exercise of the training loop on a stub model.

Covers what a run depends on but no unit test sees in isolation: that a
checkpoint holds the trained parameters only, that an interrupted run can pick
one up and continue, that the selected checkpoint is the one that comes back at
the end, and that the metric history on disk matches what was logged.
"""

from json import loads
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import degeneration_probe.probes.linear_probe as probe_module
from degeneration_probe.config import ExperimentConfig
from degeneration_probe.data.dataset import degeneration_collate_fn
from degeneration_probe.probes.linear_probe import DegenerationProbe
from degeneration_probe.training.arguments import build_training_arguments
from degeneration_probe.training.recording import METRICS_FILE, RunRecorder
from degeneration_probe.training.trainer import ProbeTrainer

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
HIDDEN = 4


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, HIDDEN)
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN), nn.Linear(HIDDEN, HIDDEN)])

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


class TinyRollouts(Dataset):
    """Four tokens per rollout, degenerate from the middle onward."""

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        # Half the rollouts degenerate, so a rank metric is defined.
        positive = index % 2 == 0
        return {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "targets": torch.tensor([0.0, 0.0, 1.0, 1.0])
            if positive
            else torch.zeros(4),
            "target_mask": torch.ones(4, dtype=torch.bool),
            "is_positive": positive,
            # Where the loop starts, which is what the rule measures coverage
            # around. A positive rollout without one cannot be measured at all.
            "onset_position": 2 if positive else None,
            "prompt_length": 0,
            "pad_token_id": 0,
            "prompt_id": f"p{index}",
            "rollout_idx": 0,
            "domain": "d",
            "split": "train",
        }


def _frozen_backbone() -> TinyModel:
    model = TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(probe_module, "get_model_layers", lambda model: list(model.layers))
    monkeypatch.setattr(probe_module, "get_model_hidden_size", lambda model: HIDDEN)
    return DegenerationProbe(_frozen_backbone(), layer_idx=1, normalization="layernorm")


@pytest.fixture
def config() -> ExperimentConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="main",
            overrides=[
                "training.runtime.max_steps=4",
                "training.runtime.gradient_accumulation_steps=2",
                "training.runtime.logging_steps=1",
                "training.validation.strategy=steps",
                "training.validation.steps=2",
                "training.checkpoint.strategy=steps",
                "training.checkpoint.steps=2",
            ],
        )
    return ExperimentConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))


def _trainer(probe, config, run_dir: Path) -> ProbeTrainer:
    dataset = TinyRollouts()
    return ProbeTrainer(
        model=probe,
        cfg=config.training,
        args=build_training_arguments(
            config, run_dir=run_dir, run_name="a_run", report_to=[], use_cpu=True
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        validation_dataset=dataset,
        final_evaluation_datasets={"val": dataset},
        pos_weight=1.0,
        data_collator=degeneration_collate_fn,
        callbacks=[RunRecorder(run_dir)],
    )


def test_a_run_checkpoints_resumes_and_reports(probe, config, tmp_path):
    trainer = _trainer(probe, config, tmp_path)
    trainer.train()

    checkpoints = sorted(path.name for path in tmp_path.glob("checkpoint-*"))
    assert checkpoints == ["checkpoint-2", "checkpoint-4"]

    saved = tmp_path / "checkpoint-4"
    # The trained parameters, the state needed to continue, and nothing else:
    # no copy of the frozen backbone.
    assert (saved / "probe.bin").is_file()
    assert (saved / "probe_config.json").is_file()
    assert (saved / "optimizer.pt").is_file()
    assert (saved / "trainer_state.json").is_file()
    assert not (saved / "model.safetensors").exists()
    assert (saved / "probe.bin").stat().st_size < 100_000

    assert trainer.state.best_model_checkpoint is not None
    assert trainer.state.best_metric is not None

    records = [
        loads(line)
        for line in (tmp_path / METRICS_FILE).read_text().splitlines()
        if line.strip()
    ]
    logged = {key for record in records for key in record}
    # The collapse guard and the class-weight diagnostic ride along with the
    # regular training record rather than in records of their own.
    assert {"loss", "train/prediction_std", "train/pos_weight"} <= logged
    assert "val/loss" in logged
    assert all("eval_loss" not in record for record in records)


def test_resuming_continues_from_the_checkpoint_rather_than_restarting(
    probe, config, tmp_path
):
    _trainer(probe, config, tmp_path).train()
    weights_at_step_two = torch.load(
        tmp_path / "checkpoint-2" / "probe.bin", weights_only=True
    )["weight"]

    fresh = DegenerationProbe(_frozen_backbone(), layer_idx=1, normalization="layernorm")
    resumed = _trainer(fresh, config, tmp_path)
    assert not torch.allclose(fresh.linear.weight, weights_at_step_two)

    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    assert resumed.state.global_step == 4
    # The history of the interrupted run came back with it, so the budget was
    # continued rather than spent again from zero.
    assert [record for record in resumed.state.log_history if record.get("step", 0) <= 2]
    assert torch.allclose(resumed.model.linear.weight, fresh.linear.weight)


def test_final_evaluation_is_namespaced_away_from_the_monitoring_curve(
    probe, config, tmp_path
):
    trainer = _trainer(probe, config, tmp_path)
    trainer.train()
    metrics = trainer.evaluate_final()
    assert "final/val/loss" in metrics
    assert "val/loss" not in metrics
