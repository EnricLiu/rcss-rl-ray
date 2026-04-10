"""Referee configuration."""

from ._base import SchemaModel


class RefereeSchema(SchemaModel):
    """Referee toggle. When *enable* is True the built-in simulation referee is active."""

    enable: bool = True
