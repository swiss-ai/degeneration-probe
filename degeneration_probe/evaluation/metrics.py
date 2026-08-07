"""Extension interface for optional validation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Type


class ValidationMetric(ABC):
    """Stateful metric updated once per validation batch."""

    @abstractmethod
    def update(self, **batch) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        raise NotImplementedError


_REGISTRY: Dict[str, Type[ValidationMetric]] = {}


def register_validation_metric(name: str, metric: Type[ValidationMetric]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Validation metric {name!r} is already registered")
    _REGISTRY[name] = metric


def build_validation_metrics(names: Iterable[str]) -> Dict[str, ValidationMetric]:
    result: Dict[str, ValidationMetric] = {}
    for name in names:
        if name not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY)) or "none"
            raise ValueError(f"Unknown validation metric {name!r}; registered metrics: {available}")
        result[name] = _REGISTRY[name]()
    return result
