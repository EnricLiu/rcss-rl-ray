"""Simulation room schema with initial state and stopping conditions."""

from pydantic import Field

from ._base import SchemaModel

class StoppingEvents(SchemaModel):
    """Conditions that end a simulation episode.

    Attributes:
        time_up: Maximum timestep; the episode is truncated when reached.
        goal_l: Goal limit for the left team; None means no limit.
        goal_r: Goal limit for the right team; None means no limit.
    """

    time_up: int | None = Field(default=None, ge=0, le=65535)
    goal_l: int | None = Field(default=None, ge=0, le=255)
    goal_r: int | None = Field(default=None, ge=0, le=255)

