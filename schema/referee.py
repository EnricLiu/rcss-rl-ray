"""Referee configuration."""

from dataclasses import dataclass


@dataclass
class RefereeSchema:
    """Referee toggle. When *enable* is True the built-in simulation referee is active."""

    enable: bool = False
