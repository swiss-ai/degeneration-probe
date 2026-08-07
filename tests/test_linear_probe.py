from types import SimpleNamespace

import torch
import torch.nn as nn

import degeneration_probe.probes.linear_probe as probe_module
from degeneration_probe.probes.linear_probe import DegenerationProbe


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.output = nn.Linear(4, 32)

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
        return SimpleNamespace(logits=self.output(hidden))


def test_scalar_linear_probe_forward_backward_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_module, "get_model_layers", lambda model: list(model.layers))
    monkeypatch.setattr(probe_module, "get_model_hidden_size", lambda model: 4)
    model = TinyModel()
    probe = DegenerationProbe(
        model,
        layer_idx=1,
        context_window_size=1,
        probe_dtype="float32",
        normalization="layernorm",
    )
    input_ids = torch.tensor([[1, 2, 3]])
    logits = probe(input_ids=input_ids)["probe_logits"]
    assert logits.shape == (1, 3)
    assert probe.linear.out_features == 1
    logits.sum().backward()
    assert probe.linear.weight.grad is not None

    probe.save(tmp_path)
    reloaded = DegenerationProbe(TinyModel(), path=tmp_path)
    assert reloaded.linear.out_features == 1
    assert torch.allclose(probe.linear.weight, reloaded.linear.weight)
