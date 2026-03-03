"""Player configuration schema."""

from .policy import Policy
from .position import Position
from dataclasses import dataclass


@dataclass
class PlayerInitState:
    """Override for a player's initial state.

    Attributes:
        pos: Normalised initial position (x, y both in [0, 1]).
        stamina: Initial stamina value, defaults to 8000.
    """

    pos: Position
    stamina: int = 8000


@dataclass
class PlayerSchema[PolicyType: Policy]:
    """Configuration for a single player; generic over the policy type.

    Attributes:
        unum: Uniform number (1-11).
        policy: Policy instance controlling this player (Bot / Agent, etc.).
        goalie: Whether this player is the goalkeeper.
        init_state: Optional initial-state override (position, stamina).
        blocklist: Optional action blocklist; keys are action names, values indicate disabled.
    """

    unum: int
    policy: PolicyType
    goalie: bool = False
    init_state: PlayerInitState | None = None
    blocklist: dict[str, bool] | None = None
