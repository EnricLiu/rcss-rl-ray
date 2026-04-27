from __future__ import annotations

import sys
from ipaddress import IPv4Address
from pathlib import Path
from types import SimpleNamespace

import gymnasium
import numpy as np
import pytest
from rcss_env.action_mask import ActionMaskResolver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.base.allocator.config import AllocatorConfig
from config import EnvConfig, ServerConfig
from rcss_env.action import Action
from rcss_env.env import RCSSEnv
from schema import (
    BotPolicy,
    GameServerSchema,
    PlayerSchema,
    PolicyAgentKind,
    PolicyKind,
    RefereeSchema,
    SspAgentPolicy,
    StoppingEvents,
    TeamsSchema,
    TeamSchema,
    TeamSide,
)


class FakeServicer:
    def __init__(self, unums: set[int]) -> None:
        self._unums = set(unums)
        self.register_calls: list[int] = []
        self.reset_calls = 0

    @property
    def unums(self) -> frozenset[int]:
        return frozenset(self._unums)

    def register(self, unum: int) -> None:
        self._unums.add(unum)
        self.register_calls.append(unum)

    def reset(self) -> None:
        self.reset_calls += 1

    def fetch_truth_world_model(self, cycle: int, timeout: float = 180.0) -> SimpleNamespace:
        return SimpleNamespace(cycle=cycle)


class RaisingAllocator:
    def request_room(self, schema: GameServerSchema):
        raise RuntimeError("allocator unavailable")


class StaticAllocator:
    def __init__(self, room: object) -> None:
        self.room = room
        self.requested_schemas: list[GameServerSchema] = []

    def request_room(self, schema: GameServerSchema):
        self.requested_schemas.append(schema)
        return self.room


class FakeRoom:
    def __init__(self, name: str = "room-a") -> None:
        self.info = SimpleNamespace(name=name)
        self.release_calls = 0
        self.started = False
        self.rcss = SimpleNamespace(trainer=SimpleNamespace(start=self._start))

    def _start(self) -> None:
        self.started = True

    def release(self, *, force: bool = True) -> None:
        self.release_calls += 1


@pytest.fixture()
def env_config() -> EnvConfig:
    return EnvConfig(
        room=GameServerSchema(
            teams=TeamsSchema(
                left=TeamSchema(
                    name="agents",
                    side=TeamSide.LEFT,
                    players=[
                        PlayerSchema(
                            unum=1,
                            policy=SspAgentPolicy(
                                kind=PolicyKind.Agent,
                                image="Cyrus2D/SoccerSimulationProxy",
                                agent=PolicyAgentKind.Ssp,
                                grpc_host=IPv4Address("127.0.0.1"),
                                grpc_port=50051,
                            ),
                        ),
                        PlayerSchema(
                            unum=2,
                            policy=SspAgentPolicy(
                                kind=PolicyKind.Agent,
                                image="Cyrus2D/SoccerSimulationProxy",
                                agent=PolicyAgentKind.Ssp,
                                grpc_host=IPv4Address("127.0.0.1"),
                                grpc_port=50051,
                            ),
                        ),
                    ],
                ),
                right=TeamSchema(
                    name="bots",
                    side=TeamSide.RIGHT,
                    players=[
                        PlayerSchema(
                            unum=1,
                            policy=BotPolicy(kind=PolicyKind.Bot, image="HELIOS/helios-base"),
                        ),
                        PlayerSchema(
                            unum=2,
                            policy=BotPolicy(kind=PolicyKind.Bot, image="HELIOS/helios-base"),
                        ),
                    ],
                ),
            ),
            stopping=StoppingEvents(time_up=100),
            referee=RefereeSchema(enable=True),
            log=False,
        ),
        grpc=ServerConfig(host=IPv4Address("127.0.0.1"), port=50051),
        allocator=AllocatorConfig(base_url="http://allocator:5555"),
    )


def _build_env(
    monkeypatch: pytest.MonkeyPatch,
    env_config: EnvConfig,
    *,
    allocator: object,
    servicer: FakeServicer,
) -> RCSSEnv:
    monkeypatch.setattr(Action, "n_actions", classmethod(lambda cls: len(cls.action_names())))

    def fake_setup(self: RCSSEnv) -> None:
        self._RCSSEnv__allocator = allocator
        self._RCSSEnv__servicer = servicer

    monkeypatch.setattr(RCSSEnv, "_setup", fake_setup)
    monkeypatch.setattr(RCSSEnv, "_start_grpc_server", lambda self: None)
    return RCSSEnv(env_config)


def test_reset_clears_episode_state_before_allocator_request_failure(
    monkeypatch: pytest.MonkeyPatch,
    env_config: EnvConfig,
) -> None:
    env = _build_env(
        monkeypatch,
        env_config,
        allocator=RaisingAllocator(),
        servicer=FakeServicer({1, 2}),
    )

    setattr(env, "_RCSSEnv__step_count", 17)
    setattr(env, "_RCSSEnv__prev_states", {1: object()})
    setattr(env, "_RCSSEnv__curr_states", {1: object()})

    with pytest.raises(gymnasium.error.ResetNeeded):
        env.reset()

    assert getattr(env, "_RCSSEnv__step_count") == 0
    assert getattr(env, "_RCSSEnv__prev_states") == {}
    assert getattr(env, "_RCSSEnv__curr_states") == {}
    assert env.has_room() is False


def test_reset_re_registers_missing_agent_unums_after_servicer_reset(
    monkeypatch: pytest.MonkeyPatch,
    env_config: EnvConfig,
) -> None:
    room = FakeRoom()
    servicer = FakeServicer({1})
    env = _build_env(
        monkeypatch,
        env_config,
        allocator=StaticAllocator(room),
        servicer=servicer,
    )

    states = {
        1: SimpleNamespace(world_model=SimpleNamespace(cycle=11)),
        2: SimpleNamespace(world_model=SimpleNamespace(cycle=11)),
    }
    monkeypatch.setattr(env, "_RCSSEnv__collect_states", lambda timeout_s=30.0: states)
    monkeypatch.setattr(
        env,
        "_RCSSEnv__states_to_obs",
        lambda payload: {
            unum: {
                "obs": np.zeros((1,), dtype=np.float32),
                ActionMaskResolver.OBSERVATION_KEY: np.ones((1,), dtype=np.int8),
            }
            for unum in payload
        },
    )

    obs, infos = env.reset()

    assert servicer.reset_calls == 1
    assert servicer.unums == frozenset({1, 2})
    assert servicer.register_calls == [2]
    assert getattr(env, "_RCSSEnv__curr_states") == states
    assert sorted(obs.keys()) == [1, 2]
    assert sorted(infos.keys()) == [1, 2]


def test_reset_releases_room_when_initial_state_collection_fails(
    monkeypatch: pytest.MonkeyPatch,
    env_config: EnvConfig,
) -> None:
    room = FakeRoom()
    env = _build_env(
        monkeypatch,
        env_config,
        allocator=StaticAllocator(room),
        servicer=FakeServicer({1, 2}),
    )

    monkeypatch.setattr(
        env,
        "_RCSSEnv__collect_states",
        lambda timeout_s=30.0: (_ for _ in ()).throw(gymnasium.error.ResetNeeded("state timeout")),
    )

    with pytest.raises(gymnasium.error.ResetNeeded):
        env.reset()

    assert room.started is True
    assert room.release_calls == 1
    assert env.has_room() is False
    assert getattr(env, "_RCSSEnv__step_count") == 0
    assert getattr(env, "_RCSSEnv__prev_states") == {}
    assert getattr(env, "_RCSSEnv__curr_states") == {}


def test_reset_syncs_runtime_grpc_port_into_allocator_schema(
    monkeypatch: pytest.MonkeyPatch,
    env_config: EnvConfig,
) -> None:
    room = FakeRoom()
    allocator = StaticAllocator(room)
    env_config.grpc.port = 0
    for player in env_config.room.teams.agent_team.players:
        if isinstance(player.policy, SspAgentPolicy):
            player.policy.grpc_port = 0

    env = _build_env(
        monkeypatch,
        env_config,
        allocator=allocator,
        servicer=FakeServicer({1, 2}),
    )

    def fake_start_grpc_server(self: RCSSEnv) -> None:
        self.config.grpc.port = 43123
        self._RCSSEnv__sync_room_grpc_port(43123)

    monkeypatch.setattr(RCSSEnv, "_start_grpc_server", fake_start_grpc_server)
    monkeypatch.setattr(
        env,
        "_RCSSEnv__collect_states",
        lambda timeout_s=30.0: {
            1: SimpleNamespace(world_model=SimpleNamespace(cycle=11)),
            2: SimpleNamespace(world_model=SimpleNamespace(cycle=11)),
        },
    )
    monkeypatch.setattr(
        env,
        "_RCSSEnv__states_to_obs",
        lambda payload: {
            unum: {
                "obs": np.zeros((1,), dtype=np.float32),
                ActionMaskResolver.OBSERVATION_KEY: np.ones((1,), dtype=np.int8),
            }
            for unum in payload
        },
    )

    env.reset()

    assert allocator.requested_schemas
    requested_schema = allocator.requested_schemas[-1]
    requested_ports = {
        player.policy.grpc_port
        for player in requested_schema.teams.agent_team.players
        if isinstance(player.policy, SspAgentPolicy)
    }
    assert requested_ports == {43123}




