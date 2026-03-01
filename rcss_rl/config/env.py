from dataclasses import dataclass, field

from .team import TeamsConfig
from .referee import RefereeConfig
from .position import Position


@dataclass
class EnvInitState:
    ball: Position = None
    timestep: int = 0

    def __post_init__(self):
        if self.timestep < 0 or self.timestep > 6000:
            raise ValueError("timestep must be in the range [0, 6000]")

@dataclass
class StoppingEvents:
    time_up: int = None
    goal_limit_l: int = None
    goal_limit_r: int = None

@dataclass
class EnvConfig:
    """Parameters that control the RCSS environment.

    . _template.json:
       https://github.com/EnricLiu/rcss_cluster/blob/sidecar/match_composer/sidecars/match_composer/docs/template.json

    Attributes
    ----------
    teams : TeamsConfig
        Configuration for the left and right teams (players, policies, etc).
    referee : RefereeConfig
        Referee settings (e.g. whether to enable the referee or not).
    init_state : EnvInitState
        Optional initial state overrides for the environment (ball position, starting timestep).
    """

    # --- Team composition (mirrors template.json) ---

    teams: TeamsConfig
    referee: RefereeConfig = field(default_factory=RefereeConfig)
    init_state: EnvInitState = None

