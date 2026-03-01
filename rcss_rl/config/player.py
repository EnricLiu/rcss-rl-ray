from .policy import Policy
from .position import Position
from dataclasses import dataclass


@dataclass
class PlayerInitState:
    """Initial state overrides for a single player.

    Matches the ``init_state`` block in the per-player section of
    *template.json*.

    Attributes
    ----------
    pos : Position
        Normalised X position (0–1 range).
    stamina : int
        Starting stamina (default ``8000``).
    """

    pos: Position
    stamina: int = 8000

@dataclass
class PlayerConfig[PolicyType: Policy]:
    """Configuration for a single player in a match.

    Mirrors the per-player block in the *template.json* accepted by the
    rcss_cluster match-composer sidecar.

    Attributes
    ----------
    unum : int
        Uniform number (1-11).
    goalie : bool
        Whether this player is a goalie.
    policy : Policy
        ``BotPolicy`` for a scripted agent or ``AgentPolicy`` for an RL training
        agent (``SspAgentPolicy`` for SoccerSimulationProxy).

    init_state : PlayerInitState | None
        Optional initial state overrides (position, stamina).
    blocklist : dict[str, bool] | None
        Optional map of action names to blocked status
        (e.g. ``{"dash": True, "catch": False}``).
    """

    unum: int
    policy: PolicyType
    goalie: bool = False
    init_state: PlayerInitState | None = None
    blocklist: dict[str, bool] | None = None
