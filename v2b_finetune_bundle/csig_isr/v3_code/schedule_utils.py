"""Learning-rate schedule helpers kept independent for regression testing."""
from __future__ import annotations
import math


def lr_multiplier(step: int, *, schedule: str, horizon: int, floor: float = 0.05) -> float:
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"unknown schedule: {schedule}")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0,1]")
    t = min(max(float(step) / float(horizon), 0.0), 1.0)
    return float(floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))
