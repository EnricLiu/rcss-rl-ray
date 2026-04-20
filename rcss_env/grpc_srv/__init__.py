"""gRPC service sub-package for communicating with SoccerSimulationProxy sidecars."""

from importlib import import_module

_EXPORTS = {
	"BatchQueue": (".batch_queue", "BatchQueue"),
	"GameServicer": (".servicer", "GameServicer"),
	"pb2": (".proto", "pb2"),
	"pb2_grpc": (".proto", "pb2_grpc"),
	"serve": (".servicer", "serve"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
	if name not in _EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module_name, attr_name = _EXPORTS[name]
	value = getattr(import_module(module_name, __name__), attr_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted(set(globals()) | set(_EXPORTS))
