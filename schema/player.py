from .policy import Policy
from .position import Position
from dataclasses import dataclass

@dataclass
class PlayerInitState:

    pos: Position
    stamina: int = 8000

@dataclass
class PlayerSchema[PolicyType: Policy]:

    unum: int
    policy: PolicyType
    goalie: bool = False
    init_state: PlayerInitState | None = None
    blocklist: dict[str, bool] | None = None
