import re
import logging

from typing import Generator
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelPathInfo:
    unum: int
    name: str
    version: str
    path: Path

    __REGEX = re.compile(r"^(.*?)-(\d+)-v(.+)$")

    @classmethod
    def parse(cls, path: Path) -> 'ModelPathInfo':
        path_name = path.root
        matches = cls.__REGEX.match(path_name)
        if matches is None:
            raise ValueError(f"Path name {path_name} does not match")

        values = matches.groups()
        if len(values) != 3:
            raise ValueError(f"Path name {path_name} does not match")

        unum = int(values[0])
        name = values[1]
        version = values[2]

        return cls(unum, name, version)


def parse_model_paths(path: Path | list[Path]) -> dict[int, ModelPathInfo]:
    def yield_from_root_path(root: Path) -> Generator[Path, None, None]:
        if not root.is_dir(): raise ValueError(f"Path {root} is not a directory")
        for p in root.iterdir():
            yield p

    if isinstance(path, Path):
        paths = yield_from_root_path(path)
    elif isinstance(path, list):
        paths = path
    else:
        raise ValueError(f"Path {path} is not a path or a list")

    ret = {}

    for model_path in paths:
        info = ModelPathInfo.parse(model_path)

        if info.unum in ret:
            logger.warning(
                f"Model path {model_path} has the same unum as {ret[info.unum]}, skipped. the expected format is <model_name>-<version>")
        else:
            ret[info.unum] = info

    return ret
