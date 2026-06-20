"""Compatibility entry point for ``python -m inference.export``."""

from .exporter import main


if __name__ == "__main__":
    raise SystemExit(main())
