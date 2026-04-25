"""Environment configuration combining room schema, gRPC, and allocator settings."""

from __future__ import annotations

from train.curriculum import CurriculumMixin

from pydantic.dataclasses import dataclass

from schema import GameServerSchema
from utils.config import ServerConfig
from client.base.allocator.config import AllocatorConfig

from .bhv import NeckViewBhv
from .reward import RewardFnMixin, DummyRewardFn

@dataclass
class EnvConfig:
    """Full configuration required by :class:`RCSSEnv`.

    Attributes:
        grpc: gRPC server settings for SoccerSimulationProxy sidecar connections.
        allocator: REST connection settings for the room allocator service.
    """

    grpc: ServerConfig
    allocator: AllocatorConfig
    curriculum: CurriculumMixin

    bhv: NeckViewBhv = NeckViewBhv()
    reward: RewardFnMixin = DummyRewardFn()
