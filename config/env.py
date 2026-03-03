"""Environment configuration combining room schema, gRPC, and allocator settings."""

from __future__ import annotations
from dataclasses import dataclass

from schema import RoomSchema
from .server import ServerConfig


@dataclass
class EnvConfig:
    """Full configuration required by :class:`RCSSEnv`.

    Attributes:
        room: Room schema (teams, stopping conditions, referee, etc.).
        grpc: gRPC server settings for SoccerSimulationProxy sidecar connections.
        allocator: REST connection settings for the room allocator service.
    """

    room: RoomSchema
    grpc: ServerConfig
    allocator: ServerConfig
