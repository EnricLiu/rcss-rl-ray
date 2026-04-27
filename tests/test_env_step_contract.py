from __future__ import annotations

import importlib
import sys
import types
import warnings
from dataclasses import dataclass
from ipaddress import IPv4Address
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.base.allocator.config import AllocatorConfig
from rcss_env.action import Action
from rcss_env.action_mask import ActionMaskResolver
from rcss_env.bhv import NeckViewBhv
from rcss_env.grpc_srv.proto import pb2
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
from utils.config import ServerConfig


class FakeServicer:
    def register(self, unum: int) -> None:
        return None

    def debug_snapshot(self) -> dict[str, Any]:
        return {"registered": []}


def _load_env_types(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Any], type[Any], type[Any]]:
    config_module = types.ModuleType("rcss_env.config")

    @dataclass
    class EnvConfig:
        grpc: ServerConfig
        allocator: AllocatorConfig
        curriculum: Any
        bhv: NeckViewBhv = NeckViewBhv()
        reward: DummyRewardFn = DummyRewardFn()

    config_module.EnvConfig = EnvConfig

    train_module = types.ModuleType("train")
    train_module.__path__ = []
    curriculum_module = types.ModuleType("train.curriculum")

    class CurriculumMixin:
        def make_schema(self) -> GameServerSchema:
            raise NotImplementedError

        def reward_fn(self) -> DummyRewardFn:
            raise NotImplementedError

    curriculum_module.CurriculumMixin = CurriculumMixin
    train_module.curriculum = curriculum_module

    monkeypatch.setitem(sys.modules, "rcss_env.config", config_module)
    monkeypatch.setitem(sys.modules, "train", train_module)
    monkeypatch.setitem(sys.modules, "train.curriculum", curriculum_module)
    sys.modules.pop("rcss_env.env", None)

    env_module = importlib.import_module("rcss_env.env")
    return EnvConfig, env_module.RCSSEnv, CurriculumMixin


def _build_schema() -> GameServerSchema:
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
                            grpc_host=IPv4Address("127.0.0.1"),
                            grpc_port=50051,
                        ),
                    ),
                    PlayerSchema(
                        unum=2,
                        policy=SspAgentPolicy(
                            image="Cyrus2D/SoccerSimulationProxy",
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
                    PlayerSchema(unum=1, policy=BotPolicy(image="HELIOS/helios-base")),
                    PlayerSchema(unum=2, policy=BotPolicy(image="HELIOS/helios-base")),
                ],
            ),
        ),
        stopping=StoppingEvents(time_up=100),
        referee=RefereeSchema(enable=True),
        log=False,
    )


def test_step_reset_needed_returns_contract_compliant_truncated_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    EnvConfig, RCSSEnv, CurriculumMixin = _load_env_types(monkeypatch)
    schema = _build_schema()

    class StaticCurriculum(CurriculumMixin):
        def make_schema(self) -> GameServerSchema:
            return schema

        def reward_fn(self) -> DummyRewardFn:
            return DummyRewardFn()

    def fake_setup(self: Any) -> None:
        self._RCSSEnv__allocator = object()
        self._RCSSEnv__servicer = FakeServicer()

    monkeypatch.setattr(Action, "n_actions", classmethod(lambda cls: len(cls.action_names())))
    monkeypatch.setattr(RCSSEnv, "_setup", fake_setup)

    env = RCSSEnv(
        EnvConfig(
            grpc=ServerConfig(host=IPv4Address("127.0.0.1"), port=50051),
            allocator=AllocatorConfig(base_url="http://allocator:5555"),
            curriculum=StaticCurriculum(),
        )
    )
    setattr(env, "_RCSSEnv__timestep", 17)

    def raise_reset_needed(_: dict[int, Any]) -> Any:
        raise gymnasium.error.ResetNeeded("state timeout")

    monkeypatch.setattr(env, "_RCSSEnv__step", raise_reset_needed)

    obs, rewards, terminateds, truncateds, infos = env.step({})

    assert sorted(obs.keys()) == [1, 2]
    for agent_id, agent_obs in obs.items():
        assert env.observation_spaces[agent_id].contains(agent_obs)
        np.testing.assert_array_equal(agent_obs["obs"], np.zeros((env.obs_dim,), dtype=np.float32))
        np.testing.assert_array_equal(
            agent_obs[ActionMaskResolver.OBSERVATION_KEY],
            Action.full_action_mask(),
        )

    assert rewards == {1: 0.0, 2: 0.0}
    assert terminateds == {1: False, 2: False, "__all__": False}
    assert truncateds == {1: True, 2: True, "__all__": True}
    assert infos == {
        1: {
            "step": 17,
            "reset_needed": True,
            "error_type": "ResetNeeded",
            "error_message": "state timeout",
        },
        2: {
            "step": 17,
            "reset_needed": True,
            "error_type": "ResetNeeded",
            "error_message": "state timeout",
        },
    }


def test_env_init_does_not_emit_box_precision_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    EnvConfig, RCSSEnv, CurriculumMixin = _load_env_types(monkeypatch)
    schema = _build_schema()

    class StaticCurriculum(CurriculumMixin):
        def make_schema(self) -> GameServerSchema:
            return schema

        def reward_fn(self) -> DummyRewardFn:
            return DummyRewardFn()

    def fake_setup(self: Any) -> None:
        self._RCSSEnv__allocator = object()
        self._RCSSEnv__servicer = FakeServicer()

    monkeypatch.setattr(Action, "n_actions", classmethod(lambda cls: len(cls.action_names())))
    monkeypatch.setattr(RCSSEnv, "_setup", fake_setup)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RCSSEnv(
            EnvConfig(
                grpc=ServerConfig(host=IPv4Address("127.0.0.1"), port=50051),
                allocator=AllocatorConfig(base_url="http://allocator:5555"),
                curriculum=StaticCurriculum(),
            )
        )

    precision_warnings = [
        str(warning.message)
        for warning in (caught or [])
        if "precision lowered" in str(warning.message)
    ]
    assert precision_warnings == []


def test_states_to_obs_coerces_float64_and_non_finite_values_into_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    EnvConfig, RCSSEnv, CurriculumMixin = _load_env_types(monkeypatch)
    env_module = sys.modules["rcss_env.env"]
    schema = _build_schema()

    class StaticCurriculum(CurriculumMixin):
        def make_schema(self) -> GameServerSchema:
            return schema

        def reward_fn(self) -> DummyRewardFn:
            return DummyRewardFn()

    def fake_setup(self: Any) -> None:
        self._RCSSEnv__allocator = object()
        self._RCSSEnv__servicer = FakeServicer()

    monkeypatch.setattr(Action, "n_actions", classmethod(lambda cls: len(cls.action_names())))
    monkeypatch.setattr(RCSSEnv, "_setup", fake_setup)

    env = RCSSEnv(
        EnvConfig(
            grpc=ServerConfig(host=IPv4Address("127.0.0.1"), port=50051),
            allocator=AllocatorConfig(base_url="http://allocator:5555"),
            curriculum=StaticCurriculum(),
        )
    )

    raw_obs = np.zeros((env.obs_dim,), dtype=np.float64)
    raw_obs[0] = np.nan
    raw_obs[1] = np.inf
    raw_obs[2] = -np.inf

    monkeypatch.setattr(env_module.observation, "extract", lambda wm: raw_obs.copy())
    monkeypatch.setattr(
        env,
        "_RCSSEnv__action_mask",
        lambda unum: np.array([1.0, np.nan, -3.0, np.inf, 0.0], dtype=np.float64),
    )

    states = {
        1: SimpleNamespace(world_model=object()),
        2: SimpleNamespace(world_model=object()),
    }

    obs = env._RCSSEnv__states_to_obs(states)

    for agent_id, agent_obs in obs.items():
        assert agent_obs["obs"].dtype == np.float32
        assert np.isfinite(agent_obs["obs"]).all()
        assert agent_obs[ActionMaskResolver.OBSERVATION_KEY].dtype == np.int8
        np.testing.assert_array_equal(
            agent_obs[ActionMaskResolver.OBSERVATION_KEY],
            np.array([1, 0, 0, 1, 0], dtype=np.int8),
        )
        assert env.observation_spaces[agent_id].contains(agent_obs)


def test_calc_reward_uses_coach_truth_instead_of_player_full_world_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    EnvConfig, RCSSEnv, CurriculumMixin = _load_env_types(monkeypatch)
    schema = _build_schema()

    class RecordingReward(DummyRewardFn):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[pb2.WorldModel | None, pb2.WorldModel]] = []

        def compute(
            self,
            prev_obs: pb2.WorldModel | None,
            prev_truth: pb2.WorldModel | None,
            curr_obs: pb2.WorldModel,
            curr_truth: pb2.WorldModel,
        ) -> float:
            self.calls.append((prev_truth, curr_truth))
            assert prev_truth is not None
            return float(curr_truth.our_team_score - prev_truth.our_team_score)

    reward = RecordingReward()

    class StaticCurriculum(CurriculumMixin):
        def make_schema(self) -> GameServerSchema:
            return schema

        def reward_fn(self) -> RecordingReward:
            return reward

    def fake_setup(self: Any) -> None:
        self._RCSSEnv__allocator = object()
        self._RCSSEnv__servicer = FakeServicer()

    monkeypatch.setattr(Action, "n_actions", classmethod(lambda cls: len(cls.action_names())))
    monkeypatch.setattr(RCSSEnv, "_setup", fake_setup)

    env = RCSSEnv(
        EnvConfig(
            grpc=ServerConfig(host=IPv4Address("127.0.0.1"), port=50051),
            allocator=AllocatorConfig(base_url="http://allocator:5555"),
            curriculum=StaticCurriculum(),
        )
    )

    prev_state = pb2.State(
        world_model=pb2.WorldModel(cycle=1, our_team_score=0),
        full_world_model=pb2.WorldModel(cycle=1, our_team_score=100),
    )
    curr_state = pb2.State(
        world_model=pb2.WorldModel(cycle=2, our_team_score=0),
        full_world_model=pb2.WorldModel(cycle=2, our_team_score=100),
    )
    prev_truth = pb2.WorldModel(cycle=1, our_team_score=2)
    curr_truth = pb2.WorldModel(cycle=2, our_team_score=3)

    reward_value = env._RCSSEnv__calc_reward(
        1,
        prev_state,
        curr_state,
        prev_truth,
        curr_truth,
    )

    assert reward_value == 1.0
    assert reward.calls[0][0].our_team_score == 2
    assert reward.calls[0][1].our_team_score == 3
