from importlib import import_module
from typing import TYPE_CHECKING

from .base.rcss.client import RcssTrainerClient

if TYPE_CHECKING:
	from .base.allocator.client import AllocatorClient
	from .base.allocator.config import AllocatorConfig
	from .base.mc.client import MatchComposerClient
	from .base.mc.config import MatchComposerConfig
	from .base.rcss.client import RcssClient, RcssClient as RcssServerClient
	from .base.rcss.config import RcssConfig
	from .fleet import FleetClient, FleetInfo
	from .room import RoomClient, RoomInfo

_EXPORTS = {
	"AllocatorClient": (".base.allocator", "AllocatorClient"),
	"AllocatorConfig": (".base.allocator", "AllocatorConfig"),
	"FleetClient": (".fleet", "FleetClient"),
	"FleetInfo": (".fleet", "FleetInfo"),
	"MatchComposerClient": (".base.mc", "MatchComposerClient"),
	"MatchComposerConfig": (".base.mc", "MatchComposerConfig"),
	"RcssClient": (".base.rcss", "RcssClient"),
	"RcssConfig": (".base.rcss", "RcssConfig"),
	"RcssServerClient": (".base.rcss", "RcssServerClient"),
	"RcssTrainerClient": (".base.rcss", "RcssTrainerClient"),
	"RoomClient": (".room", "RoomClient"),
	"RoomInfo": (".room", "RoomInfo"),
}

__all__ = [
	"AllocatorClient",
	"FleetClient",
	"FleetInfo",
	"MatchComposerClient",
	"RcssClient",
	"RcssTrainerClient",
	"RoomClient",
	"RoomInfo",
]


def __getattr__(name: str):
	if name not in _EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module_name, attr_name = _EXPORTS[name]
	value = getattr(import_module(module_name, __name__), attr_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted(set(globals()) | set(_EXPORTS))


