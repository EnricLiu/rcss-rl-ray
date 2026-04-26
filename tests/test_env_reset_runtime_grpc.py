from __future__ import annotations

from ipaddress import IPv4Address
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from client.base.allocator.config import AllocatorConfig
from rcss_env.action_mask import ActionMaskResolver
from rcss_env.action import Action
from rcss_env.config import EnvConfig
from rcss_env.env import RCSSEnv
from rcss_env.reward import DummyRewardFn
from schema import (
    BotPolicy,
    GameServerSchema,
    PlayerSchema,
    RefereeSchema,
    SspAgentPolicy,
    StoppingEvents,
    TeamsSchema,
    TeamSchema,
    TeamSide,
)
from train.curriculum import CurriculumMixin
from utils.config import ServerConfig


class FakeServicer:
    def __init__(self, unums: set[int]) -> None:
        self._unums = set(unums)

    @property
    def unums(self) -> frozenset[int]:
        return frozenset(self._unums)

    def register(self, unum: int) -> None:
        self._unums.add(unum)

    def reset(self) -> None:
        return None

    def debug_snapshot(self) -> dict[str, Any]:
        return {"registered": sorted(self._unums)}


class FakeAllocator:
    def __init__(self) -> None:
        self.requested_schemas: list[GameServerSchema] = []
        self.rooms: list[FakeRoom] = []

    def request_room(self, schema: GameServerSchema) -> "FakeRoom":
        self.requested_schemas.append(schema.model_copy(deep=True))
        room = FakeRoom(name=f"room-{len(self.requested_schemas)}")
        self.rooms.append(room)
        return room


class FakeRoom:
    def __init__(self, name: str) -> None:
        self.info = SimpleNamespace(
            name=name,
            base_url_rcss="http://rcss",
            base_url_mc="http://mc",
        )
        self.release_calls = 0
        self.started = False
        self.rcss = SimpleNamespace(trainer=SimpleNamespace(start=self._start))

    def _start(self) -> None:
        self.started = True

    def release(self) -> None:
        self.release_calls += 1


class StaticCurriculum(CurriculumMixin):
    def __init__(self, grpc_server: ServerConfig) -> None:
        self.config = SimpleNamespace(grpc_server=grpc_server)
        self._reward = DummyRewardFn()

    def make_schema(self) -> GameServerSchema:
        grpc_server = self.config.grpc_server
        return GameServerSchema(
            teams=TeamsSchema(
                left=TeamSchema(
                    name="agents",
                    side=TeamSide.LEFT,
                    players=[
                        PlayerSchema(
                            unum=1,
                            policy=SspAgentPolicy(
                                image="Cyrus2D/SoccerSimulationProxy",
                                grpc_host=grpc_server.host,
                                grpc_port=grpc_server.port,
                            ),
                        ),
                        PlayerSchema(
                            unum=2,
                            policy=SspAgentPolicy(
                                image="Cyrus2D/SoccerSimulationProxy",
                                grpc_host=grpc_server.host,
                                grpc_port=grpc_server.port,
                            ),
                        ),
                    ],
                ),
                right=TeamSchema(
                    name="bots",
                    side=TeamSide.RIGHT,
                    players=[
                        PlayerSchema(unum=1, policy=BotPolicy(image="HELIOS/helios-base")),
                        PlayerSchema(unum=2, policy=BotPolicy(image="HELIOS/helios-base")),
                    ],
                ),
            ),
            stopping=StoppingEvents(time_up=100),
            referee=RefereeSchema(enable=True),
            log=False,
        )

    def reward_fn(self) -> DummyRewardFn:
        return self._reward


@pytest.fixture()
def env_config() -> EnvConfig:
    grpc_server = ServerConfig(host=IPv4Address("0.0.0.0"), port=0)
    curriculum = StaticCurriculum(grpc_server=grpc_server)
    return EnvConfig(
        grpc=grpc_server.model_copy(deep=True),
        allocator=AllocatorConfig(base_url="http://allocator:5555"),
        curriculum=curriculum,
    )


def test_reset_resyncs_runtime_grpc_endpoint_for_new_schema(
    monkeypatch: pytest.MonkeyPatch,
    env_config: EnvConfig,
) -> None:
    allocator = FakeAllocator()
    servicer = FakeServicer({1, 2})
    serve_calls: list[int] = []

    def fake_setup(self: RCSSEnv) -> None:
        self._RCSSEnv__allocator = allocator
        self._RCSSEnv__servicer = servicer

    monkeypatch.setattr(RCSSEnv, "_setup", fake_setup)

    env = RCSSEnv(env_config)

    monkeypatch.setattr("rcss_env.env.get_node_ip_address", lambda: "127.0.0.1")
    fake_states = {
        1: SimpleNamespace(world_model=SimpleNamespace(cycle=11)),
        2: SimpleNamespace(world_model=SimpleNamespace(cycle=11)),
    }
    monkeypatch.setattr(
        env,
        "_RCSSEnv__collect_states",
        lambda timeout_s=180.0: fake_states,
    )
    monkeypatch.setattr(
        env,
        "_RCSSEnv__states_to_obs",
        lambda payload: {
            unum: {
                "obs": np.zeros((env.obs_dim,), dtype=np.float32),
                ActionMaskResolver.OBSERVATION_KEY: np.ones((Action.n_actions(),), dtype=np.int8),
            }
            for unum in payload
        },
    )

    def tracked_serve(servicer: Any, port: int, block: bool) -> tuple[object, int, object]:
        serve_calls.append(port)
        return object(), 43123, object()

    monkeypatch.setattr("rcss_env.grpc_srv.servicer.serve", tracked_serve)

    env.reset()
    env.reset()

    assert serve_calls == [0]
    assert len(allocator.requested_schemas) == 2

    for requested_schema in allocator.requested_schemas:
        requested_endpoints = {
            (str(player.policy.grpc_host), player.policy.grpc_port)
            for player in requested_schema.teams.agent_team.players
            if isinstance(player.policy, SspAgentPolicy)
        }
        assert requested_endpoints == {("127.0.0.1", 43123)}

    assert str(env.config.grpc.host) == "127.0.0.1"
    assert env.config.grpc.port == 43123
    curriculum = env_config.curriculum
    assert isinstance(curriculum, StaticCurriculum)
    assert str(curriculum.config.grpc_server.host) == "127.0.0.1"
    assert curriculum.config.grpc_server.port == 43123

