"""Data types for the degeneration probe pipeline."""

from dataclasses import dataclass


@dataclass
class DegenerationItem:
    """A single labelled generation for probe training/evaluation."""

    prompt: str
    completion: str
    label: float  # 1.0 = degenerate, 0.0 = normal
