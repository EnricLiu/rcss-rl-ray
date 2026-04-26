from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .allocator import AllocatorClient, AllocatorConfig
	from .mc import MatchComposerClient, MatchComposerConfig
	from .rcss import RcssClient, RcssConfig, RcssServerClient, RcssTrainerClient

_EXPORTS = {
	"AllocatorClient": (".allocator", "AllocatorClient"),
	"AllocatorConfig": (".allocator", "AllocatorConfig"),
	"MatchComposerClient": (".mc", "MatchComposerClient"),
	"MatchComposerConfig": (".mc", "MatchComposerConfig"),
	"RcssClient": (".rcss", "RcssClient"),
	"RcssConfig": (".rcss", "RcssConfig"),
	"RcssServerClient": (".rcss", "RcssServerClient"),
	"RcssTrainerClient": (".rcss", "RcssTrainerClient"),
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
