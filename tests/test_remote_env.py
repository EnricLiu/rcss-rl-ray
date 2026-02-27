"""Unit tests for RCSSEnv remote-mode components."""

from __future__ import annotations

import pytest

from rcss_rl.config import EnvConfig, PlayerConfig
from rcss_rl.env.rcss_env import RCSSEnv
from rcss_rl.proto import service_pb2 as pb2


class TestParseAgentId:
    def test_left_side(self) -> None:
        side, unum = RCSSEnv._parse_agent_id("left_3")
        assert side == "left"
        assert unum == 3

    def test_right_side(self) -> None:
        side, unum = RCSSEnv._parse_agent_id("right_11")
        assert side == "right"
        assert unum == 11


class TestRemoteModeAgentIds:
    def test_agent_ids_only_agents(self) -> None:
        """In remote mode, only policy_kind='agent' players are RL agents."""
        cfg = EnvConfig(
            mode="remote",
            allocator_url="http://fake:6000",
            ally_players=[
                PlayerConfig(unum=1, goalie=True, policy_kind="bot"),
                PlayerConfig(unum=2, goalie=False, policy_kind="agent"),
                PlayerConfig(unum=3, goalie=False, policy_kind="agent"),
            ],
            opponent_players=[
                PlayerConfig(unum=1, goalie=True, policy_kind="bot"),
            ],
        )
        env = RCSSEnv(cfg)
        assert env._agent_ids == {"left_2", "left_3"}
        assert env._ally_agent_unums == [2, 3]
        assert env._opponent_agent_unums == []

    def test_agent_ids_both_sides(self) -> None:
        """Agents on both sides are registered."""
        cfg = EnvConfig(
            mode="remote",
            allocator_url="http://fake:6000",
            ally_players=[
                PlayerConfig(unum=2, policy_kind="agent"),
            ],
            opponent_players=[
                PlayerConfig(unum=5, policy_kind="agent"),
            ],
        )
        env = RCSSEnv(cfg)
        assert env._agent_ids == {"left_2", "right_5"}

    def test_local_mode_all_ids(self) -> None:
        """In local mode, all players get position-based IDs."""
        cfg = EnvConfig(
            mode="local",
            ally_players=[
                PlayerConfig(unum=1, policy_kind="bot"),
                PlayerConfig(unum=2, policy_kind="agent"),
            ],
            opponent_players=[
                PlayerConfig(unum=1, policy_kind="bot"),
            ],
        )
        env = RCSSEnv(cfg)
        # Local mode: all 3 players are controllable.
        assert env._agent_ids == {"left_0", "left_1", "right_0"}


class TestActionToProto:
    @pytest.fixture()
    def env(self) -> RCSSEnv:
        """A remote-mode env (we only need _action_to_proto)."""
        cfg = EnvConfig(
            mode="remote",
            allocator_url="http://fake:6000",
            ally_players=[PlayerConfig(unum=2, policy_kind="agent")],
            opponent_players=[],
        )
        return RCSSEnv(cfg)

    def test_noop_has_neck_only(self, env: RCSSEnv) -> None:
        pa = env._action_to_proto(0)
        assert isinstance(pa, pb2.PlayerActions)
        # NOOP → only neck scan action.
        assert len(pa.actions) == 1
        assert pa.actions[0].HasField("neck_turn_to_ball_or_scan")

    def test_dash_forward(self, env: RCSSEnv) -> None:
        pa = env._action_to_proto(1)
        assert len(pa.actions) == 2
        assert pa.actions[0].HasField("dash")
        assert pa.actions[0].dash.power == 100.0

    def test_dash_backward(self, env: RCSSEnv) -> None:
        pa = env._action_to_proto(2)
        assert pa.actions[0].dash.power == -100.0

    def test_turn_left(self, env: RCSSEnv) -> None:
        pa = env._action_to_proto(3)
        assert pa.actions[0].HasField("turn")
        assert pa.actions[0].turn.relative_direction == -30.0

    def test_turn_right(self, env: RCSSEnv) -> None:
        pa = env._action_to_proto(4)
        assert pa.actions[0].turn.relative_direction == 30.0

    def test_kick(self, env: RCSSEnv) -> None:
        pa = env._action_to_proto(5)
        assert pa.actions[0].HasField("kick")
        assert pa.actions[0].kick.power == 100.0


class TestBuildRoomRequest:
    def test_room_request_matches_config(self) -> None:
        cfg = EnvConfig(
            mode="remote",
            allocator_url="http://fake:6000",
            ally_team_name="MyTeam",
            opponent_team_name="TheirTeam",
            ally_players=[
                PlayerConfig(unum=2, policy_kind="agent", policy_agent="ssp",
                             policy_image="Cyrus2D/SSP",
                             grpc_host="10.0.0.1", grpc_port=50051),
            ],
            opponent_players=[
                PlayerConfig(unum=1, goalie=True, policy_kind="bot",
                             policy_image="HELIOS/helios-base"),
            ],
            time_up=3000,
            goal_limit_l=5,
            referee_enable=True,
            ball_init_x=0.5,
            ball_init_y=0.5,
        )
        env = RCSSEnv(cfg)
        req = env._build_room_request()
        d = req.to_dict()

        assert d["api_version"] == 1
        assert d["stopping"]["time_up"] == 3000
        assert d["stopping"]["goal_l"] == 5
        assert d["referee"]["enable"] is True
        assert d["init_state"]["ball"]["x"] == 0.5
        assert d["teams"]["allies"]["name"] == "MyTeam"
        assert d["teams"]["opponents"]["name"] == "TheirTeam"
        assert d["teams"]["allies"]["players"][0]["policy"]["kind"] == "agent"
        assert d["teams"]["opponents"]["players"][0]["policy"]["kind"] == "bot"


class TestRemoteObsDim:
    """Remote obs_dim should match the _world_model_to_obs layout."""

    def test_remote_obs_dim_3v3(self) -> None:
        cfg = EnvConfig(
            mode="remote",
            allocator_url="http://fake:6000",
            ally_players=[
                PlayerConfig(unum=i, policy_kind="agent") for i in range(1, 4)
            ],
            opponent_players=[
                PlayerConfig(unum=i, policy_kind="bot") for i in range(1, 4)
            ],
        )
        env = RCSSEnv(cfg)
        # ball(4) + self(7) + teammates(2)*4 + opponents(3)*4 + match(4) = 35
        assert env._obs_dim == 35
        assert env.observation_space.shape == (35,)

    def test_remote_obs_dim_1v1(self) -> None:
        cfg = EnvConfig(
            mode="remote",
            allocator_url="http://fake:6000",
            ally_players=[PlayerConfig(unum=1, policy_kind="agent")],
            opponent_players=[PlayerConfig(unum=1, policy_kind="bot")],
        )
        env = RCSSEnv(cfg)
        # ball(4) + self(7) + teammates(0)*4 + opponents(1)*4 + match(4) = 19
        assert env._obs_dim == 19
        assert env.observation_space.shape == (19,)

    def test_local_obs_dim_unchanged(self) -> None:
        cfg = EnvConfig(
            ally_players=[
                PlayerConfig(unum=1, goalie=True),
                PlayerConfig(unum=2),
            ],
            opponent_players=[
                PlayerConfig(unum=1, goalie=True),
                PlayerConfig(unum=2),
            ],
        )
        env = RCSSEnv(cfg)
        # local: 8 + (4-1)*2 + 4 = 18
        assert env._obs_dim == 18
        assert env.observation_space.shape == (18,)


class TestWorkerVectorIndexExtraction:
    """worker_index / vector_index are extracted from dict configs."""

    def test_default_indices(self) -> None:
        env = RCSSEnv()
        assert env._worker_index == 0
        assert env._vector_index == 0

    def test_dict_config_with_indices(self) -> None:
        cfg = {
            "worker_index": 3,
            "vector_index": 2,
            "ally_players": [
                {"unum": 1, "goalie": True, "policy_kind": "agent"},
            ],
            "opponent_players": [
                {"unum": 1, "goalie": True, "policy_kind": "bot"},
            ],
        }
        env = RCSSEnv(cfg)
        assert env._worker_index == 3
        assert env._vector_index == 2

    def test_dict_config_without_indices(self) -> None:
        cfg = {
            "ally_players": [
                {"unum": 1, "goalie": True, "policy_kind": "agent"},
            ],
            "opponent_players": [
                {"unum": 1, "goalie": True, "policy_kind": "bot"},
            ],
        }
        env = RCSSEnv(cfg)
        assert env._worker_index == 0
        assert env._vector_index == 0
