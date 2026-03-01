from enum import Enum
from typing import Generator, Any
from dataclasses import dataclass

from .player import PlayerConfig
from .policy import Policy, PolicyKind, PolicyAgentKind, BotPolicy, AgentPolicy, SspAgentPolicy

class TeamSide(Enum):
    LEFT = "left"
    RIGHT = "right"

@dataclass
class TeamConfig:
    name: str
    side: TeamSide
    players: list[PlayerConfig[Policy]]

    def __post_init__(self):
        if len(self.name) > 10:
            raise ValueError("Team name must be <= 10 characters")

        if len(self.players) > 11:
            raise ValueError("A team cannot have more than 11 players")


    def agents(self) -> Generator[PlayerConfig[AgentPolicy], Any, None]:
        for p in self.players:
            if p.policy.kind == PolicyKind.Agent:
                if isinstance(p.policy, AgentPolicy):
                    p: PlayerConfig[AgentPolicy]
                    yield p
                else:
                    raise ValueError(f"Player {p.unum} has policy kind 'agent' but policy is not an AgentPolicy instance")

    def ssp_agents(self) -> Generator[PlayerConfig[SspAgentPolicy], Any, list[Any]]:
        ret = []


        for p in self.agents():
            if p.policy.agent == PolicyAgentKind.Ssp:
                if isinstance(p.policy, SspAgentPolicy):
                    p: PlayerConfig[SspAgentPolicy]
                    yield p

        return ret

    def bots(self) -> Generator[PlayerConfig[BotPolicy], Any, None]:
        for p in self.players:
            if p.policy.kind == PolicyKind.Bot:
                if isinstance(p.policy, BotPolicy):
                    p: PlayerConfig[BotPolicy]
                    yield p
                else:
                    raise ValueError(
                        f"Player {p.unum} has policy kind 'agent' but policy is not an AgentPolicy instance")


    def is_agentic(self) -> bool:
        try:
            next(self.agents())
            return True
        except StopIteration:
            return False

@dataclass
class TeamsConfig:
    left: TeamConfig
    right: TeamConfig