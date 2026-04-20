"""Schema module defining simulation room, team, player, and policy data structures."""

from .team import TeamsSchema, TeamSchema, TeamSide
from .player import PlayerSchema, PlayerInitState, PlayerActionList
from .policy import Policy, PolicyKind, PolicyAgentKind, BotPolicy, AgentPolicy, SspAgentPolicy
from .referee import RefereeSchema
from .position import Position
from .stopping import StoppingEvents
from .init_state import RoomInitState

from pydantic import AliasChoices, Field

from ._base import SchemaModel

SCHEMA_VERSION: int = 1

class GameServerSchema(SchemaModel):
    """Parameters that control the RCSS environment.

    . _template.json:
       https://github.com/EnricLiu/rcss_cluster/blob/sidecar/match_composer/sidecars/match_composer/docs/template.json

    Attributes:
        teams: Left and right team configurations.
        stopping: Stopping-condition settings.
        referee: Referee settings.
        init_state: Optional initial-state overrides.
        log: Whether to enable logging for the processes in the simulation.
    """
    teams: TeamsSchema
    referee: RefereeSchema = Field(default_factory=RefereeSchema, validation_alias=AliasChoices("referee", "referees"))
    stopping: StoppingEvents = Field(default_factory=StoppingEvents)
    init_state: RoomInitState = Field(default_factory=RoomInitState)
    env: dict[str, str] | None = None
    log: bool = False

