from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .config import GenDatasetCurriculumConfig


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
        raise ValueError("Dataset generation config file must contain a mapping at the top level")
    return loaded


def load_gen_dataset_config(path: str | Path) -> GenDatasetCurriculumConfig:
    return GenDatasetCurriculumConfig.model_validate(load_config_mapping(path))
