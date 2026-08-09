"""What value each token is trained toward.

Three families, all producing the same shape of output: a per-token target and
a mask marking positions with no defined target. That uniform shape is what
lets token selection stay unaware of which family is in use. Selection decides
*which* tokens enter a batch, labelling decides *what value* each one is
trained toward, and neither needs to know about the other.

The families are:

``frontier_hard``
    A step at the degeneration frontier, shifted by a horizon N:
    ``y(t) = 1`` exactly when ``f_r - t <= N``. At ``N = 0`` this is the plain
    statement "degenerate from the frontier onward". Larger N asks the probe to
    fire N tokens before degeneration is confirmed. No special case is needed
    past the frontier: the distance simply goes negative there and the
    condition stays satisfied.

``frontier_soft``
    The same frontier with the step replaced by a decay, because the hard
    label claims a token one position before the frontier is as innocent as one
    a thousand positions before, which nobody believes. The decay length says
    how far back the run-up is taken to reach.

``token_signal``
    A per-token quantity computed independently of the frontier, such as a
    repetition score or the model's own predictive entropy. These define a
    target everywhere, including on rollouts that ended cleanly, where a
    nonzero score describes text that is repetitive but legitimate. They
    therefore teach a somewhat different concept, which is why the choice of
    target is an experimental axis rather than a settled question.

A family returns ``None`` for a rollout it cannot label, and the caller drops
it rather than inventing a value.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

FRONTIER_FAMILIES = {"frontier_hard", "frontier_soft"}
LABEL_FAMILIES = FRONTIER_FAMILIES | {"token_signal"}
TOKEN_SIGNALS = {"repetition_score", "entropy"}
DECAY_SHAPES = {"linear", "exponential"}


def _frontier_of(stop_reason: str, onset_position: Optional[float], num_tokens: int) -> Optional[int]:
    """The frontier for a rollout, or None when it has no usable one."""
    if stop_reason == "eos":
        return None
    if onset_position is None or pd.isna(onset_position):
        return None
    onset = int(onset_position)
    if onset < 0 or onset >= num_tokens:
        raise ValueError(f"onset_position={onset} is outside a {num_tokens}-token rollout")
    return onset


def frontier_hard_targets(
    num_tokens: int,
    *,
    stop_reason: str,
    onset_position: Optional[float],
    horizon: int = 0,
) -> Optional[List[float]]:
    """A step at the frontier, brought forward by ``horizon`` tokens."""
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if stop_reason == "eos":
        return [0.0] * num_tokens
    onset = _frontier_of(stop_reason, onset_position, num_tokens)
    if onset is None:
        return None
    start = max(onset - horizon, 0)
    return [0.0] * start + [1.0] * (num_tokens - start)


def frontier_soft_targets(
    num_tokens: int,
    *,
    stop_reason: str,
    onset_position: Optional[float],
    decay: str = "exponential",
    decay_length: float = 256.0,
) -> Optional[List[float]]:
    """One inside the pattern, decaying to zero back through the run-up."""
    if decay not in DECAY_SHAPES:
        raise ValueError(f"decay must be one of {sorted(DECAY_SHAPES)}, got {decay!r}")
    if decay_length <= 0:
        raise ValueError(f"decay_length must be positive, got {decay_length}")
    if stop_reason == "eos":
        return [0.0] * num_tokens
    onset = _frontier_of(stop_reason, onset_position, num_tokens)
    if onset is None:
        return None
    positions = np.arange(num_tokens, dtype=np.float64)
    distance = np.maximum(onset - positions, 0.0)
    if decay == "exponential":
        targets = np.exp(-distance / decay_length)
    else:
        targets = np.clip(1.0 - distance / decay_length, 0.0, 1.0)
    targets[onset:] = 1.0
    return targets.tolist()


def token_signal_targets(values: Sequence[float], num_tokens: int) -> List[float]:
    """Align a per-token signal to the token ids; unmatched positions stay masked."""
    targets = [float("nan")] * num_tokens
    for index, value in enumerate(list(values)[:num_tokens]):
        if value is not None and not pd.isna(value):
            targets[index] = float(value)
    return targets


def derive_targets(
    label_config,
    *,
    num_tokens: int,
    stop_reason: str,
    onset_position: Optional[float],
    signal: Optional[Sequence[float]] = None,
) -> Optional[List[float]]:
    """Produce one rollout's targets under the configured family."""
    family = label_config.family
    if family == "frontier_hard":
        return frontier_hard_targets(
            num_tokens,
            stop_reason=stop_reason,
            onset_position=onset_position,
            horizon=label_config.horizon,
        )
    if family == "frontier_soft":
        return frontier_soft_targets(
            num_tokens,
            stop_reason=stop_reason,
            onset_position=onset_position,
            decay=label_config.decay,
            decay_length=label_config.decay_length,
        )
    if family == "token_signal":
        if signal is None:
            raise ValueError(
                f"label family 'token_signal' needs the {label_config.signal!r} column"
            )
        return token_signal_targets(signal, num_tokens)
    raise ValueError(f"Unknown label family {family!r}; expected one of {sorted(LABEL_FAMILIES)}")


def produces_binary_targets(label_config) -> bool:
    """Whether every defined target is exactly 0 or 1.

    Class weighting and the corpus positive rate are only meaningful for a
    family that answers yes; asking for them elsewhere is a configuration
    mistake rather than something to approximate.
    """
    return label_config.family == "frontier_hard"


def required_signal(label_config) -> Optional[str]:
    """Which per-token column, if any, the configured family needs loaded."""
    return label_config.signal if label_config.family == "token_signal" else None
