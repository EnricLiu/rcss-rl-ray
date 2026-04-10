from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from ._base import SchemaModel
from .policy import BotPolicy, Policy, SspAgentPolicy
from .position import Position


class PlayerInitState(SchemaModel):
    pos: Position | None = None
    stamina: int | None = Field(default=None, ge=0, le=65535)


class PlayerActionList(SchemaModel):
    dash: bool = False
    catch_: bool = Field(default=False, alias="catch", serialization_alias="catch")


class PlayerSchema(SchemaModel):
    """Configuration for a single player; generic over the policy type.

        Attributes:
            unum: Uniform number (1-11).
            policy: Policy instance controlling this player (Bot / Agent, etc.).
            goalie: Whether this player is the goalkeeper.
            init_state: Optional initial-state override (position, stamina).
            blocklist: Optional action blocklist; keys are action names, values indicate disabled.
        """

    unum: int = Field(ge=1, le=11)
    goalie: bool = False
    policy: BotPolicy | SspAgentPolicy = Field(default_factory=Policy.helios_base)
    init_state: PlayerInitState = Field(default_factory=PlayerInitState)
    blocklist: PlayerActionList = Field(default_factory=PlayerActionList)

    @field_validator("policy", mode="before")
    @classmethod
    def _parse_policy(cls, value: Any) -> BotPolicy | SspAgentPolicy:
        return Policy.parse(value)
