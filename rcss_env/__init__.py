"""RCSS multi-agent reinforcement learning environment package.

Public API:
- GameServicer / pb2: gRPC service implementation and protobuf message definitions
- AllocatorClient: Room allocator REST client
- RCSSEnv: Ray/RLlib-compatible multi-agent Gymnasium environment
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from client.base.allocator import AllocatorClient
	from .env import RCSSEnv
	from .grpc_srv.batch_queue import BatchQueue
	from .grpc_srv.proto import pb2, pb2_grpc
	from .grpc_srv.servicer import GameServicer, serve

_EXPORTS = {
	"AllocatorClient": ("client.base.allocator", "AllocatorClient"),
	"BatchQueue": (".grpc_srv", "BatchQueue"),
	"GameServicer": (".grpc_srv", "GameServicer"),
	"RCSSEnv": (".env", "RCSSEnv"),
	"pb2": (".grpc_srv", "pb2"),
	"pb2_grpc": (".grpc_srv", "pb2_grpc"),
	"serve": (".grpc_srv", "serve"),
}

__all__ = ["AllocatorClient", "GameServicer", "RCSSEnv", "pb2"]


def __getattr__(name: str):
	if name not in _EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module_name, attr_name = _EXPORTS[name]
	value = getattr(import_module(module_name, __name__), attr_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted(set(globals()) | set(_EXPORTS))
