"""Normalised 2-D coordinate in the range [0, 1]."""

from pydantic import field_validator

from ._base import SchemaModel

X_RANGE: tuple[float, float] = (-52.5, 52.5)
Y_RANGE: tuple[float, float] = (-34.0, 34.0)


class Position(SchemaModel):
    """Normalised pitch position. Both x and y must be in [0.0, 1.0]."""

    x: float
    y: float

    @field_validator("x")
    @classmethod
    def _validate_x_interval(cls, value: float, info) -> float:
        x_low, x_high = X_RANGE
        if not (x_low <= value <= x_high):
            raise ValueError(f"{info.field_name} must be in the range [{x_low}, {x_high}]")
        return value

    @field_validator("y")
    @classmethod
    def _validate_y_interval(cls, value: float, info) -> float:
        y_low, y_high = Y_RANGE
        if not (y_low <= value <= y_high):
            raise ValueError(f"{info.field_name} must be in the range [{y_low}, {y_high}]")
        return value
