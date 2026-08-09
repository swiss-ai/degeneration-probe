import json
from types import SimpleNamespace

import torch
import torch.nn as nn

import degeneration_probe.probes.linear_probe as probe_module
from degeneration_probe.probes.linear_probe import DegenerationProbe
from degeneration_probe.training.recording import (
    HISTORY_FILE,
    METRICS_FILE,
    RUN_INFO_FILE,
    RunInfo,
    RunRecorder,
    prepare_run_location,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

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


def _state(step, epoch=1.0):
    return SimpleNamespace(global_step=step, epoch=epoch, is_world_process_zero=True)


def test_every_logged_record_is_mirrored_locally(tmp_path):
    recorder = RunRecorder(tmp_path)
    recorder.on_log(None, _state(10), None, logs={"loss": 0.5, "train/prediction_std": 0.1})
    recorder.on_log(None, _state(10), None, logs={"val/loss": 0.4})
    recorder.on_log(None, _state(20), None, logs={"loss": 0.3})

    lines = (tmp_path / METRICS_FILE).read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["train/prediction_std"] == 0.1

    recorder.write_history()
    import pandas as pd

    history = pd.read_parquet(tmp_path / HISTORY_FILE)
    # One row per step, with the training and validation records merged.
    assert list(history["step"]) == [10, 20]
    assert history.loc[history["step"] == 10, "val/loss"].item() == 0.4


def test_an_interrupted_run_still_has_a_readable_history(tmp_path):
    recorder = RunRecorder(tmp_path)
    recorder.on_log(None, _state(1), None, logs={"loss": 1.0})
    assert (tmp_path / METRICS_FILE).is_file()


def test_run_info_records_identity_and_outcome(tmp_path):
    info = RunInfo(
        run_name="a_run",
        group="a_group",
        tags=["loss:bce"],
        axes={"loss": "bce"},
        run_dir=str(tmp_path),
        attempt="20260808T110000",
    )
    info.write(tmp_path)
    assert json.loads((tmp_path / RUN_INFO_FILE).read_text())["status"] == "running"

    info.finish(tmp_path, status="failed", error="boom")
    written = json.loads((tmp_path / RUN_INFO_FILE).read_text())
    assert written["status"] == "failed"
    assert written["error"] == "boom"
    assert written["duration_seconds"] is not None
    assert written["environment"]["versions"]["torch"]


def test_probe_weights_survive_a_save_and_reload_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_module, "get_model_layers", lambda model: list(model.layers))
    monkeypatch.setattr(probe_module, "get_model_hidden_size", lambda model: 4)
    probe = DegenerationProbe(TinyModel(), layer_idx=1, normalization="layernorm")
    probe.save(tmp_path)
    trained = probe.linear.weight.detach().clone()

    with torch.no_grad():
        probe.linear.weight.add_(1.0)
    assert not torch.allclose(probe.linear.weight, trained)

    probe.load_weights(tmp_path)
    assert torch.allclose(probe.linear.weight, trained)


def _finished(run_dir, status="finished"):
    RunInfo(
        run_name="r",
        group="g",
        tags=[],
        axes={},
        run_dir=str(run_dir),
        attempt=run_dir.name,
    ).finish(run_dir, status=status)


def test_repeating_a_run_never_overwrites_the_one_before_it(tmp_path):
    first = prepare_run_location(tmp_path, "a_run", "auto")
    _finished(first.run_dir)
    second = prepare_run_location(tmp_path, "a_run", "auto")

    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == second.run_dir.parent == tmp_path / "a_run"
    assert second.continued is False
    assert second.resume_from is None
    # Attempt names sort in the order they happened.
    assert sorted([first.attempt, second.attempt]) == [first.attempt, second.attempt]


def test_the_latest_attempt_is_reachable_by_a_stable_path(tmp_path):
    prepare_run_location(tmp_path, "a_run", "auto")
    latest = prepare_run_location(tmp_path, "a_run", "auto")
    link = tmp_path / "a_run" / "latest"
    assert link.is_symlink()
    assert link.resolve() == latest.run_dir.resolve()


def test_an_interrupted_attempt_is_continued_in_place(tmp_path):
    first = prepare_run_location(tmp_path, "a_run", "auto")
    _finished(first.run_dir, status="failed")
    (first.run_dir / "checkpoint-2").mkdir()
    RunInfo(
        run_name="r",
        group="g",
        tags=[],
        axes={},
        run_dir=str(first.run_dir),
        attempt=first.attempt,
        wandb={"id": "abc123"},
    ).write(first.run_dir)

    resumed = prepare_run_location(tmp_path, "a_run", "auto")
    assert resumed.run_dir == first.run_dir
    assert resumed.continued is True
    assert resumed.resume_from.endswith("checkpoint-2")
    assert resumed.wandb_id == "abc123"


def test_a_finished_attempt_is_never_continued(tmp_path):
    first = prepare_run_location(tmp_path, "a_run", "auto")
    (first.run_dir / "checkpoint-2").mkdir()
    _finished(first.run_dir, status="finished")

    following = prepare_run_location(tmp_path, "a_run", "auto")
    assert following.run_dir != first.run_dir
    assert following.continued is False


def test_resume_never_always_starts_a_fresh_attempt(tmp_path):
    first = prepare_run_location(tmp_path, "a_run", "never")
    (first.run_dir / "checkpoint-2").mkdir()
    second = prepare_run_location(tmp_path, "a_run", "never")
    assert second.run_dir != first.run_dir
    assert second.resume_from is None


def test_an_explicit_checkpoint_starts_a_new_attempt_from_it(tmp_path):
    first = prepare_run_location(tmp_path, "a_run", "auto")
    checkpoint = first.run_dir / "checkpoint-2"
    checkpoint.mkdir()
    second = prepare_run_location(tmp_path, "a_run", str(checkpoint))
    assert second.run_dir != first.run_dir
    assert second.resume_from == str(checkpoint)


def test_attempts_started_in_the_same_second_get_separate_directories(tmp_path):
    locations = [prepare_run_location(tmp_path, "a_run", "never") for _ in range(3)]
    directories = {location.run_dir for location in locations}
    assert len(directories) == 3
    # And the newest of them is still the one an auto-resume would consider.
    (locations[-1].run_dir / "checkpoint-2").mkdir()
    assert prepare_run_location(tmp_path, "a_run", "auto").run_dir == locations[-1].run_dir
