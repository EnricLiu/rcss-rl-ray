from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rcss_env.grpc_srv.proto import pb2

from .projector import AgentIndexEntry


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_world_model(path: Path, wm: pb2.WorldModel) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = wm.SerializeToString()
    with path.open("ab") as fh:
        offset = fh.tell()
        fh.write(struct.pack(">I", len(payload)))
        fh.write(payload)
    return offset


def iter_world_models(path: Path) -> Iterable[pb2.WorldModel]:
    with path.open("rb") as fh:
        while True:
            header = fh.read(4)
            if not header:
                return
            if len(header) != 4:
                raise ValueError(f"Corrupt world-model stream header in {path}")
            size = struct.unpack(">I", header)[0]
            payload = fh.read(size)
            if len(payload) != size:
                raise ValueError(f"Corrupt world-model stream payload in {path}")
            wm = pb2.WorldModel()
            wm.ParseFromString(payload)
            yield wm


def write_agent_index(path: Path, entries: list[AgentIndexEntry]) -> None:
    write_json(path, [entry.to_json() for entry in entries])


def write_cycles(path: Path, cycles: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(cycles, dtype=np.int32))


def write_obs(path: Path, obs_rows: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if obs_rows:
        payload = np.stack(obs_rows).astype(np.float32, copy=False)
    else:
        payload = np.empty((0, 0, 144), dtype=np.float32)
    np.save(path, payload)
