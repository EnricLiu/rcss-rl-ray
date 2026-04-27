"""Cycle-indexed coach truth world-model buffer."""

from __future__ import annotations

import asyncio

from .proto import pb2


DEFAULT_MAX_TRUTH_CYCLES = 16


class TruthWorldModelTimeoutError(RuntimeError):
    """Raised when a requested coach truth world model is not available."""


class TruthWorldModelBuffer:
    """Asynchronous exact-cycle buffer for coach truth world models."""

    def __init__(self, max_cycles: int = DEFAULT_MAX_TRUTH_CYCLES) -> None:
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        self._max_cycles = max_cycles
        self._world_models: dict[int, pb2.WorldModel] = {}
        self._latest_cycle: int | None = None
        self._condition = asyncio.Condition()

    async def reset(self) -> None:
        async with self._condition:
            self._world_models.clear()
            self._latest_cycle = None
            self._condition.notify_all()

    async def put(self, world_model: pb2.WorldModel) -> None:
        cycle = int(world_model.cycle)
        copied = pb2.WorldModel()
        copied.CopyFrom(world_model)

        async with self._condition:
            self._world_models[cycle] = copied
            if self._latest_cycle is None or cycle > self._latest_cycle:
                self._latest_cycle = cycle
            self._trim_locked()
            self._condition.notify_all()

    async def get(self, cycle: int, timeout: float) -> pb2.WorldModel:
        cycle = int(cycle)

        async def _wait_for_cycle() -> pb2.WorldModel:
            async with self._condition:
                await self._condition.wait_for(lambda: cycle in self._world_models)
                copied = pb2.WorldModel()
                copied.CopyFrom(self._world_models[cycle])
                return copied

        try:
            return await asyncio.wait_for(_wait_for_cycle(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            snapshot = self.snapshot()
            raise TruthWorldModelTimeoutError(
                f"Timed out waiting for coach truth world model cycle={cycle}; "
                f"buffered_cycles={snapshot['buffered_cycles']} latest_cycle={snapshot['latest_cycle']}"
            ) from exc

    async def discard_before(self, cycle: int) -> None:
        async with self._condition:
            for stale_cycle in [c for c in self._world_models if c < cycle]:
                self._world_models.pop(stale_cycle, None)

    def snapshot(self) -> dict[str, object]:
        buffered_cycles = sorted(self._world_models)
        return {
            "latest_cycle": self._latest_cycle,
            "buffered_cycles": buffered_cycles,
            "buffer_size": len(buffered_cycles),
            "max_cycles": self._max_cycles,
        }

    def _trim_locked(self) -> None:
        overflow = len(self._world_models) - self._max_cycles
        if overflow <= 0:
            return

        for stale_cycle in sorted(self._world_models)[:overflow]:
            self._world_models.pop(stale_cycle, None)
