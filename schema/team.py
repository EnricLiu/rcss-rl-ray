"""Team configuration schema with player filtering and validation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Generator

from pydantic import ConfigDict, field_validator, model_validator

from ._base import SchemaModel
from .player import PlayerSchema
from .policy import BotPolicy, PolicyKind, SspAgentPolicy


class TeamSide(str, Enum):
    """Team side enum: LEFT or RIGHT."""
    LEFT = "left"
    RIGHT = "right"



class TeamSchema(SchemaModel):
    """Configuration for a single team.

    Attributes:
        name: Team name, at most 10 characters.
        side: Team side (LEFT / RIGHT).
        players: List of players, at most 11.
    """
    name: str
    side: TeamSide
    players: list[PlayerSchema]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("Team name cannot be empty")
        if not value.isascii():
            raise ValueError("Team name cannot contain non-ASCII characters")
        if len(value) > 16:
            raise ValueError("Team name cannot be longer than 16 characters")
        return value

    @field_validator("players")
    @classmethod
    def _validate_players(cls, value: list[PlayerSchema]) -> list[PlayerSchema]:
        if len(value) > 11:
            raise ValueError("A team cannot have more than 11 players")
        return value

    def agents(self) -> Generator[PlayerSchema, Any, None]:
        """Yield all players in this team that use an Agent policy."""
        for p in self.players:
            if p.policy.kind == PolicyKind.Agent:
                yield p

    def ssp_agents(self) -> Generator[PlayerSchema, Any, list[Any]]:
        for p in self.agents():
            if isinstance(p.policy, SspAgentPolicy):
                yield p

        return []

    def bots(self) -> Generator[PlayerSchema, Any, None]:
        """Yield all players in this team that use a Bot policy."""
        for p in self.players:
            if isinstance(p.policy, BotPolicy):
                yield p

    def is_agentic(self) -> bool:
        """Return True if the team contains at least one Agent-controlled player."""
        return any(True for _ in self.agents())


class TeamsSchema(SchemaModel):
    """Combined configuration for both teams.

    Validation rules:
    - The left team's side must be LEFT and the right team's side must be RIGHT.
    - Exactly one team must contain agent-controlled players.

    Attributes:
        left: Left team configuration.
        right: Right team configuration.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        use_enum_values=False,
    )

    left: TeamSchema
    right: TeamSchema

    @model_validator(mode="before")
    @classmethod
    def _apply_side_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        value = dict(value)
        left = value.get("left")
        if isinstance(left, dict) and "side" not in left:
            value["left"] = {**left, "side": TeamSide.LEFT}

        right = value.get("right")
        if isinstance(right, dict) and "side" not in right:
            value["right"] = {**right, "side": TeamSide.RIGHT}

        return value

    @model_validator(mode="after")
    def _validate_teams(self) -> TeamsSchema:
        if self.left.side != TeamSide.LEFT:
            raise ValueError("Left team must have side=TeamSide.LEFT")
        if self.right.side != TeamSide.RIGHT:
            raise ValueError("Right team must have side=TeamSide.RIGHT")
        if self.left.side == self.right.side:
            raise ValueError("Teams cannot be on the same side")
        if self.left.name == self.right.name:
            raise ValueError("Teams cannot share the same name")
        return self

    @property
    def agent_teams(self) -> list[TeamSchema]:
        return [team for team in (self.left, self.right) if team.is_agentic()]

    @property
    def agent_team(self) -> TeamSchema:
        """Return the team that contains agent-controlled players."""
        agent_teams = self.agent_teams
        if len(agent_teams) != 1:
            raise ValueError("RCSSEnv expects exactly one agentic team")
        return agent_teams[0]
