from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .schema import TrainConfig


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config files require PyYAML to be installed") from exc
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    elif suffix == ".toml":
        with config_path.open("rb") as fh:
            loaded = tomllib.load(fh)
    else:
        raise ValueError(f"Unsupported config file extension: {config_path.suffix!r}")

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Training config file must contain a mapping at the top level")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_train_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    payload = load_config_mapping(path)
    if overrides:
        payload = deep_merge(payload, overrides)
    return TrainConfig.model_validate(payload)

