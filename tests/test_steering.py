"""Tests for steering strategies."""

import torch
import pytest

from degeneration_probe.worker.steering import (
    SteeringContext,
    SteeringStrategy,
    TemperatureBoostStrategy,
    get_strategy,
)


def test_temperature_boost_should_intervene_above_threshold():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    assert strategy.should_intervene(probe_score=0.9, threshold=0.8) is True


def test_temperature_boost_should_not_intervene_below_threshold():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    assert strategy.should_intervene(probe_score=0.5, threshold=0.8) is False


def test_temperature_boost_should_not_intervene_at_threshold():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    assert strategy.should_intervene(probe_score=0.8, threshold=0.8) is False


def test_temperature_boost_modifies_logits():
    strategy = TemperatureBoostStrategy(boost_temperature=2.0)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ctx = SteeringContext(recent_token_ids=[], position=10)
    result = strategy.intervene(logits, ctx)
    expected = logits / 2.0
    assert torch.allclose(result, expected)


def test_temperature_boost_preserves_shape():
    strategy = TemperatureBoostStrategy(boost_temperature=1.5)
    logits = torch.randn(50257)
    ctx = SteeringContext(recent_token_ids=[], position=0)
    result = strategy.intervene(logits, ctx)
    assert result.shape == logits.shape


def test_get_strategy_returns_temperature_boost():
    strategy = get_strategy("temperature_boost", boost_temperature=2.0)
    assert isinstance(strategy, TemperatureBoostStrategy)


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("unknown_strategy")
