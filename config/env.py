"""Environment configuration combining room schema, gRPC, and allocator settings."""

from __future__ import annotations

from pydantic import BaseModel

from schema import GameServerSchema
from .server import ServerConfig
from .allocator import AllocatorConfig


class EnvConfig(BaseModel):
    """Full configuration required by :class:`RCSSEnv`.

    Attributes:
        room: GameServer schema (teams, stopping conditions, referee, etc.).
        grpc: gRPC server settings for SoccerSimulationProxy sidecar connections.
        allocator: REST connection settings for the room allocator service.
    """

    room: GameServerSchema
    grpc: ServerConfig
    allocator: AllocatorConfig
