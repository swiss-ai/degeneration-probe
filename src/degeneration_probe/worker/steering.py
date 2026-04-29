"""Steering strategies for probe-guided generation intervention."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class SteeringContext:
    """Context passed to steering strategies during intervention."""
    recent_token_ids: list[int]
    position: int


class SteeringStrategy(ABC):
    """Base class for steering strategies."""

    @abstractmethod
    def should_intervene(self, probe_score: float, threshold: float) -> bool:
        ...

    @abstractmethod
    def intervene(self, logits: torch.Tensor, context: SteeringContext) -> torch.Tensor:
        ...


class TemperatureBoostStrategy(SteeringStrategy):
    """When probe score exceeds threshold, divide logits by boost_temperature."""

    def __init__(self, boost_temperature: float = 1.5):
        self.boost_temperature = boost_temperature

    def should_intervene(self, probe_score: float, threshold: float) -> bool:
        return probe_score > threshold

    def intervene(self, logits: torch.Tensor, context: SteeringContext) -> torch.Tensor:
        return logits / self.boost_temperature


STRATEGY_REGISTRY: dict[str, type[SteeringStrategy]] = {
    "temperature_boost": TemperatureBoostStrategy,
}


def get_strategy(name: str, **kwargs) -> SteeringStrategy:
    """Instantiate a steering strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)
