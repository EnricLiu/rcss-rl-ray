"""Simulation room schema with initial state and stopping conditions."""

from dataclasses import dataclass, field

from .team import TeamsSchema
from .referee import RefereeSchema
from .position import Position


@dataclass
class RoomInitState:
    """Optional initial-state overrides for a simulation room.

    Attributes:
        ball: Normalised initial ball position; None uses the default.
        timestep: Starting simulation timestep, must be in [0, 6000].
    """

    ball: Position = None
    timestep: int = 0

    def __post_init__(self):
        if self.timestep < 0 or self.timestep > 6000:
            raise ValueError("timestep must be in the range [0, 6000]")


@dataclass
class StoppingEvents:
    """Conditions that end a simulation episode.

    Attributes:
        time_up: Maximum timestep; the episode is truncated when reached.
        goal_limit_l: Goal limit for the left team; None means no limit.
        goal_limit_r: Goal limit for the right team; None means no limit.
    """

    time_up: int = 6000
    goal_limit_l: int = None
    goal_limit_r: int = None


@dataclass
class RoomSchema:
    """Complete configuration for a simulation room.

    Attributes:
        teams: Left and right team configurations.
        stopping: Stopping-condition settings.
        referee: Referee settings.
        init_state: Optional initial-state overrides.
    """

    teams: TeamsSchema
    stopping: StoppingEvents = field(default_factory=StoppingEvents)
    referee: RefereeSchema = field(default_factory=RefereeSchema)
    init_state: RoomInitState = None
