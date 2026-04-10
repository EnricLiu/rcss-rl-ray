from __future__ import annotations

from ipaddress import IPv4Address
from enum import Enum
from typing import Any, Literal, cast

from pydantic import Field, IPvAnyAddress, field_validator

from ._base import SchemaModel


class PolicyKind(str, Enum):
    """Top-level policy kind: Bot (scripted) or Agent (RL-trained)."""
    Bot = "bot"
    Agent = "agent"


class PolicyAgentKind(str, Enum):
    """Agent policy subtype. Currently only Ssp (SoccerSimulationProxy) is supported."""
    Ssp = "ssp"


def _validate_policy_image(image: str) -> str:
    if image != "*" and "/" not in image:
        raise ValueError(r"Invalid policy name, should be in pattern /^\w+/(\w+|\*):?\w*?$/")
    return image


class Policy(SchemaModel):
    """Base policy class.

    Attributes:
        kind: Policy kind (Bot / Agent).
        image: Container image name for this policy.
    """
    kind: PolicyKind
    image: str

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: str) -> str:
        return _validate_policy_image(value)

    @staticmethod
    def helios_base() -> BotPolicy:
        return BotPolicy(image="HELIOS/helios-base")

    @staticmethod
    def parse(maybe_policy: dict[str, Any] | Policy) -> BotPolicy | SspAgentPolicy:
        if isinstance(maybe_policy, (BotPolicy, SspAgentPolicy)):
            return maybe_policy
        if isinstance(maybe_policy, Policy):
            maybe_policy = maybe_policy.model_dump()
        if not isinstance(maybe_policy, dict):
            raise TypeError("policy must be a mapping or Policy instance")

        kind = maybe_policy.get("kind")
        if kind in {PolicyKind.Bot, PolicyKind.Bot.value}:
            return BotPolicy.model_validate(maybe_policy)
        if kind in {PolicyKind.Agent, PolicyKind.Agent.value}:
            return cast(SspAgentPolicy, AgentPolicy.parse(maybe_policy))

        raise ValueError(f"Unknown policy kind: {kind}, expected one of {[e.value for e in PolicyKind]}")


class BotPolicy(Policy):
    kind: Literal[PolicyKind.Bot] = PolicyKind.Bot

    @staticmethod
    def parse(policy: Policy | dict[str, Any]) -> BotPolicy:
        if isinstance(policy, BotPolicy):
            return policy
        if isinstance(policy, Policy):
            policy = policy.model_dump()
        return BotPolicy.model_validate(policy)


class AgentPolicy(Policy):
    agent: PolicyAgentKind

    kind: Literal[PolicyKind.Agent] = PolicyKind.Agent

    @staticmethod
    def parse(policy: Policy | dict[str, Any], maybe_agent_policy: dict[str, Any] | None = None) -> SspAgentPolicy:
        payload: Any = maybe_agent_policy if maybe_agent_policy is not None else policy
        if isinstance(payload, SspAgentPolicy):
            return payload
        if isinstance(payload, Policy):
            payload = payload.model_dump()

        agent = payload.get("agent") if isinstance(payload, dict) else None
        if agent in {PolicyAgentKind.Ssp, PolicyAgentKind.Ssp.value}:
            return SspAgentPolicy.model_validate(payload)

        raise ValueError(f"Unknown agent kind: {agent}, expected one of {[e.value for e in PolicyAgentKind]}")


class SspAgentPolicy(AgentPolicy):
    agent: Literal[PolicyAgentKind.Ssp] = PolicyAgentKind.Ssp
    grpc_host: IPvAnyAddress | IPv4Address
    grpc_port: int = Field(ge=1, le=65535)

    def grpc_addr(self) -> str:
        """Return the gRPC address as a ``host:port`` string."""
        return f"{self.grpc_host}:{self.grpc_port}"

