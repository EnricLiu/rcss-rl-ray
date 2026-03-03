from __future__ import annotations

from enum import Enum
from typing import Any
from dataclasses import dataclass

class PolicyKind(Enum):
    Bot = "bot"
    Agent = "agent"

class PolicyAgentKind(Enum):
    Ssp = "ssp"

@dataclass
class Policy:
    kind: PolicyKind
    image: str

    def __post_init__(self):
        if not self.image:
            raise ValueError("image must be provided for Policy")

    @staticmethod
    def parse(maybe_policy: dict[str, Any]) -> Policy:
        kind_str = maybe_policy.get("kind")
        image = maybe_policy.get("image")

        try:
            kind = PolicyKind(kind_str)
        except Exception:
            raise ValueError(f"Unknown policy kind: {kind_str}, expected one of {[e.value for e in PolicyKind]}")

        policy = Policy(kind=kind, image=image)
        match kind:
            case PolicyKind.Bot:
                return BotPolicy.parse(policy)
            case PolicyKind.Agent:
                maybe_agent = maybe_policy
                return AgentPolicy.parse(policy, maybe_agent)

            case _:
                raise ValueError(f"Unsupported policy kind: {kind}")

@dataclass
class BotPolicy(Policy):
    def __post_init__(self):
        if self.kind != PolicyKind.Bot:
            raise ValueError("kind must be 'bot' for BotPolicy")

    @staticmethod
    def parse(policy: Policy) -> BotPolicy:
        return BotPolicy(kind=policy.kind, image=policy.image)

@dataclass
class AgentPolicy(Policy):
    agent: PolicyAgentKind
    def __post_init__(self):
        if self.kind != PolicyKind.Agent:
            raise ValueError("kind must be 'agent' for AgentPolicy")
        if not self.agent:
            raise ValueError("agent must be provided for AgentPolicy")

    @staticmethod
    def parse(policy: Policy, maybe_agent_policy: dict[str, Any]) -> AgentPolicy:
        agent_str = maybe_agent_policy.get("agent")
        try:
            agent = PolicyAgentKind(agent_str)
            match agent:
                case PolicyAgentKind.Ssp:
                    return SspAgentPolicy(
                        kind=PolicyKind.Agent,
                        image=policy.image,
                        agent=agent,
                        grpc_host=maybe_agent_policy.get("grpc_host"),
                        grpc_port=maybe_agent_policy.get("grpc_port"),
                    )
                case _:
                    raise ValueError(f"Unsupported agent kind: {agent}")

        except ValueError:
            raise ValueError(f"Unknown agent kind: {agent_str}, expected one of {[e.value for e in PolicyAgentKind]}")

@dataclass
class SspAgentPolicy(AgentPolicy):
    grpc_host: str
    grpc_port: int

    def __post_init__(self):
        if self.agent != PolicyAgentKind.Ssp:
            raise ValueError("agent must be 'ssp' for SspAgentPolicy")

        if not self.grpc_host:
            raise ValueError("grpc_host must be provided for SspAgentPolicy")
        if not self.grpc_port:
            raise ValueError("grpc_port must be provided for SspAgentPolicy")

        if self.grpc_port <= 0 or self.grpc_port > 65535:
            raise ValueError("grpc_port must be in the range 1-65535")

    def grpc_addr(self) -> str:
        return f"{self.grpc_host}:{self.grpc_port}"
