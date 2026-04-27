from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from ._base import SchemaModel
from .policy import BotPolicy, Policy, SspAgentPolicy


class CoachSchema(SchemaModel):

    policy: BotPolicy | SspAgentPolicy = Field(default_factory=Policy.helios_base)

    @field_validator("policy", mode="before")
    @classmethod
    def _parse_policy(cls, value: Any) -> BotPolicy | SspAgentPolicy:
        return Policy.parse(value)