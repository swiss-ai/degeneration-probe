"""Probe architecture for degeneration training."""

from .linear_probe import DegenerationProbe, setup_probe

__all__ = ["DegenerationProbe", "setup_probe"]
