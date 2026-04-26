from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from ipaddress import IPv4Address
import sys
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import collect_room_diagnostics, collect_runtime_snapshot
from client.room.info import RoomInfo


class FakeProbe:
    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._result


class FakeEnv:
    def __init__(self) -> None:
        self.room = SimpleNamespace(
            info=RoomInfo(
                name="room-a",
                pod=IPv4Address("10.0.0.5"),
                host=IPv4Address("127.0.0.1"),
                ports={"default": 6666, "mc": 7777},
            ),
            mc=SimpleNamespace(status=FakeProbe({"ok": True})),
            rcss=SimpleNamespace(
                metrics_status=FakeProbe({"status": "idle"}),
                metrics_health=FakeProbe({"healthy": True}),
                metrics_conn=FakeProbe({"conn": 1}),
                trainer=SimpleNamespace(
                    team_names=FakeProbe(exc=RuntimeError("trainer timed out")),
                ),
            ),
        )

    def runtime_diagnostics(self) -> dict[str, object]:
        return {"step_count": 3, "curr_state_cycles": {"1": 11, "2": 11}}


def test_collect_runtime_snapshot_does_not_hit_live_probes() -> None:
    env = FakeEnv()

    snapshot = collect_runtime_snapshot(cast(Any, env))

    assert snapshot["room"] == {
        "name": "room-a",
        "pod": "10.0.0.5",
        "host": "127.0.0.1",
        "ports": {"default": 6666, "mc": 7777},
    }
    assert snapshot["env_runtime"] == {"step_count": 3, "curr_state_cycles": {"1": 11, "2": 11}}
    assert "deferred" in snapshot["note"]
    assert env.room.mc.status.calls == 0
    assert env.room.rcss.metrics_status.calls == 0
    assert env.room.rcss.metrics_health.calls == 0
    assert env.room.rcss.metrics_conn.calls == 0
    assert env.room.rcss.trainer.team_names.calls == 0


def test_collect_room_diagnostics_records_probe_durations_and_errors() -> None:
    env = FakeEnv()

    diagnostics = collect_room_diagnostics(cast(Any, env))

    assert diagnostics["mc_status"] == {"ok": True}
    assert diagnostics["rcss_metrics_status"] == {"status": "idle"}
    assert diagnostics["rcss_metrics_health"] == {"healthy": True}
    assert diagnostics["rcss_metrics_conn"] == {"conn": 1}
    assert diagnostics["trainer_team_names"] == {
        "error_type": "RuntimeError",
        "message": "trainer timed out",
    }
    assert set(diagnostics["probe_duration_s"].keys()) == {
        "mc_status",
        "rcss_metrics_status",
        "rcss_metrics_health",
        "rcss_metrics_conn",
        "trainer_team_names",
    }
    assert env.room.mc.status.calls == 1
    assert env.room.rcss.metrics_status.calls == 1
    assert env.room.rcss.metrics_health.calls == 1
    assert env.room.rcss.metrics_conn.calls == 1
    assert env.room.rcss.trainer.team_names.calls == 1



