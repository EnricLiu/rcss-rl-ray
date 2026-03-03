"""Normalised 2-D coordinate in the range [0, 1]."""

from dataclasses import dataclass


@dataclass
class Position:
    """Normalised pitch position. Both x and y must be in [0.0, 1.0]."""

    x: float
    y: float

    def __post_init__(self):
        if not (0.0 <= self.x <= 1.0):
            raise ValueError("x must be in the range [0, 1]")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError("y must be in the range [0, 1]")
