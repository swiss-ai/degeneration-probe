"""Tests for the token-by-token generation engine."""

import pytest

from degeneration_probe.worker.engine import GenerationEngine, TokenResult


class FakeModel:
    """Minimal fake model for testing the engine without GPU."""

    class FakeConfig:
        hidden_size = 64

    def __init__(self):
        self.config = self.FakeConfig()
        self.device = "cpu"

    def __call__(self, **kwargs):
        import torch

        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        # Return a fake output with logits and hidden states
        logits = torch.randn(batch, seq_len, 100)
        return type("Output", (), {"logits": logits})()

    def parameters(self):
        return iter([])


class FakeTokenizer:
    """Minimal fake tokenizer."""

    eos_token_id = 99
    pad_token = "<pad>"

    def __call__(self, text, return_tensors=None, **kwargs):
        import torch
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, **kwargs):
        import torch
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        return type("Enc", (), {"to": lambda self, d: self, "input_ids": ids, "attention_mask": mask})()

    def decode(self, ids, **kwargs):
        return " ".join(f"tok{i}" for i in ids)


def test_token_result_fields():
    r = TokenResult(
        token_id=42,
        token_text="hello",
        position=0,
        probe_score=0.5,
        was_steered=False,
    )
    assert r.token_id == 42
    assert r.probe_score == 0.5


def test_engine_init():
    engine = GenerationEngine(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        probe=None,
    )
    assert engine.model is not None
    assert engine.tokenizer is not None
