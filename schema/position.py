"""Normalised 2-D coordinate in the range [0, 1]."""

from pydantic import field_validator

from ._base import SchemaModel



class Position(SchemaModel):
    """Normalised pitch position. Both x and y must be in [0.0, 1.0]."""

    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def _validate_unit_interval(cls, value: float, info) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{info.field_name} must be in the range [0, 1]")
        return value
