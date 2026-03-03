from __future__ import annotations
from dataclasses import dataclass

from schema import RoomSchema
from .server import ServerConfig

@dataclass
class EnvConfig:
    room: RoomSchema
    grpc: ServerConfig
    allocator: ServerConfig
