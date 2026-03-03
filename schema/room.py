from dataclasses import dataclass, field

from .team import TeamsSchema
from .referee import RefereeSchema
from .position import Position

@dataclass
class RoomInitState:
    ball: Position = None
    timestep: int = 0

    def __post_init__(self):
        if self.timestep < 0 or self.timestep > 6000:
            raise ValueError("timestep must be in the range [0, 6000]")

@dataclass
class StoppingEvents:
    time_up: int = 6000
    goal_limit_l: int = None
    goal_limit_r: int = None

@dataclass
class RoomSchema:

    teams: TeamsSchema
    stopping: StoppingEvents = field(default_factory=StoppingEvents)
    referee: RefereeSchema = field(default_factory=RefereeSchema)
    init_state: RoomInitState = None
