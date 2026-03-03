"""Schema module defining simulation room, team, player, and policy data structures."""

from .room import RoomSchema, RoomInitState, StoppingEvents
from .team import TeamsConfig, TeamConfig, TeamSide
from .player import PlayerSchema, PlayerInitState
from .policy import Policy, PolicyKind, PolicyAgentKind, BotPolicy, AgentPolicy, SspAgentPolicy
from .referee import RefereeSchema
from .position import Position
