"""Team configuration schema with player filtering and validation."""

from enum import Enum
from typing import Generator, Any
from dataclasses import dataclass

from .player import PlayerSchema
from .policy import Policy, PolicyKind, PolicyAgentKind, BotPolicy, AgentPolicy, SspAgentPolicy


class TeamSide(Enum):
    """Team side enum: LEFT or RIGHT."""

    LEFT = "left"
    RIGHT = "right"


@dataclass
class TeamSchema:
    """Configuration for a single team.

    Attributes:
        name: Team name, at most 10 characters.
        side: Team side (LEFT / RIGHT).
        players: List of players, at most 11.
    """

    name: str
    side: TeamSide
    players: list[PlayerSchema[Policy]]

    def __post_init__(self):
        if len(self.name) > 10:
            raise ValueError("Team name must be <= 10 characters")

        if len(self.players) > 11:
            raise ValueError("A team cannot have more than 11 players")

    def agents(self) -> Generator[PlayerSchema[AgentPolicy], Any, None]:
        """Yield all players in this team that use an Agent policy."""
        for p in self.players:
            if p.policy.kind == PolicyKind.Agent:
                if isinstance(p.policy, AgentPolicy):
                    p: PlayerSchema[AgentPolicy]
                    yield p
                else:
                    raise ValueError(f"Player {p.unum} has policy kind 'agent' but policy is not an AgentPolicy instance")

    def ssp_agents(self) -> Generator[PlayerSchema[SspAgentPolicy], Any, list[Any]]:
        """Yield all players in this team that use an SspAgentPolicy."""
        ret = []

        for p in self.agents():
            if p.policy.agent == PolicyAgentKind.Ssp:
                if isinstance(p.policy, SspAgentPolicy):
                    p: PlayerSchema[SspAgentPolicy]
                    yield p

        return ret

    def bots(self) -> Generator[PlayerSchema[BotPolicy], Any, None]:
        """Yield all players in this team that use a Bot policy."""
        for p in self.players:
            if p.policy.kind == PolicyKind.Bot:
                if isinstance(p.policy, BotPolicy):
                    p: PlayerSchema[BotPolicy]
                    yield p
                else:
                    raise ValueError(
                        f"Player {p.unum} has policy kind 'agent' but policy is not an AgentPolicy instance")

    def is_agentic(self) -> bool:
        """Return True if the team contains at least one Agent-controlled player."""
        try:
            next(self.agents())
            return True
        except StopIteration:
            return False


@dataclass
class TeamsSchema:
    """Combined configuration for both teams.

    Validation rules:
    - The left team's side must be LEFT and the right team's side must be RIGHT.
    - Exactly one team must contain agent-controlled players.

    Attributes:
        left: Left team configuration.
        right: Right team configuration.
    """

    left: TeamSchema
    right: TeamSchema

    def __post_init__(self):
        if self.left.side != TeamSide.LEFT:
            raise ValueError("Left team must have side=TeamSide.LEFT")
        if self.right.side != TeamSide.RIGHT:
            raise ValueError("Right team must have side=TeamSide.RIGHT")

        left_is_agentic = self.left.is_agentic()
        right_is_agentic = self.right.is_agentic()

        if left_is_agentic and right_is_agentic:
            raise ValueError("Only one team can have agent-controlled players (for self-play training, use a single team with both agent and bot players)")

        if not left_is_agentic and not right_is_agentic:
            raise ValueError("At least one team must have agent-controlled players")

        if left_is_agentic: self.__agent_team_side = TeamSide.LEFT
        else: self.__agent_team_side = TeamSide.RIGHT

    @property
    def agent_team(self):
        """Return the team that contains agent-controlled players."""
        return self.left if self.__agent_team_side == TeamSide.LEFT else self.right
