"""Unit tests for allocator_client data models."""

from __future__ import annotations

from rcss_rl.env.allocator_client import (
    PlayerConfig,
    PlayerInitState,
    RoomRequest,
)


class TestPlayerInitState:
    def test_to_dict_full(self) -> None:
        s = PlayerInitState(pos_x=0.5, pos_y=0.3, stamina=6000)
        assert s.to_dict() == {"pos": {"x": 0.5, "y": 0.3}, "stamina": 6000}

    def test_to_dict_pos_only(self) -> None:
        s = PlayerInitState(pos_x=0.1, pos_y=0.9)
        assert s.to_dict() == {"pos": {"x": 0.1, "y": 0.9}}

    def test_to_dict_stamina_only(self) -> None:
        s = PlayerInitState(stamina=4000)
        assert s.to_dict() == {"stamina": 4000}

    def test_to_dict_empty(self) -> None:
        s = PlayerInitState()
        assert s.to_dict() == {}


class TestRoomRequestToDict:
    def test_minimal_template(self) -> None:
        """Minimal RoomRequest produces correct template.json structure."""
        req = RoomRequest(
            ally_name="SSP",
            opponent_name="HB",
            ally_players=[
                PlayerConfig(
                    unum=1,
                    goalie=True,
                    policy_kind="agent",
                    policy_agent="ssp",
                    policy_image="Cyrus2D/SoccerSimulationProxy",
                    grpc_host="127.0.0.1",
                    grpc_port=6657,
                ),
            ],
            opponent_players=[
                PlayerConfig(
                    unum=1,
                    goalie=True,
                    policy_kind="bot",
                    policy_image="HELIOS/helios-base",
                ),
            ],
            time_up=6000,
        )
        d = req.to_dict()
        assert d["api_version"] == 1
        assert d["referee"] == {"enable": False}
        assert d["stopping"] == {"time_up": 6000, "goal_l": 0}
        assert "init_state" not in d  # no ball init

        ally = d["teams"]["allies"]
        assert ally["name"] == "SSP"
        p = ally["players"][0]
        assert p["unum"] == 1
        assert p["goalie"] is True
        assert p["policy"]["kind"] == "agent"
        assert p["policy"]["agent"] == "ssp"
        assert p["policy"]["image"] == "Cyrus2D/SoccerSimulationProxy"
        assert p["policy"]["grpc_host"] == "127.0.0.1"
        assert p["policy"]["grpc_port"] == 6657

        opp = d["teams"]["opponents"]
        assert opp["name"] == "HB"
        op = opp["players"][0]
        assert op["policy"]["kind"] == "bot"
        assert op["policy"]["image"] == "HELIOS/helios-base"
        assert "grpc_host" not in op["policy"]

    def test_ball_init_state(self) -> None:
        req = RoomRequest(ball_init_x=0.5, ball_init_y=0.5)
        d = req.to_dict()
        assert d["init_state"] == {"ball": {"x": 0.5, "y": 0.5}}

    def test_player_init_state_and_blocklist(self) -> None:
        req = RoomRequest(
            ally_players=[
                PlayerConfig(
                    unum=1,
                    goalie=True,
                    policy_kind="bot",
                    policy_image="HELIOS/helios-base",
                    init_state=PlayerInitState(pos_x=0.9, pos_y=0.5, stamina=6000),
                    blocklist={"dash": True, "catch": False},
                ),
            ],
        )
        d = req.to_dict()
        p = d["teams"]["allies"]["players"][0]
        assert p["init_state"] == {"pos": {"x": 0.9, "y": 0.5}, "stamina": 6000}
        assert p["blocklist"] == {"dash": True, "catch": False}

    def test_full_template_matches_reference(self) -> None:
        """Verify output matches the reference template.json structure."""
        req = RoomRequest(
            referee_enable=False,
            time_up=6000,
            goal_limit_l=0,
            ball_init_x=0.5,
            ball_init_y=0.5,
            ally_name="SSP",
            opponent_name="HB",
            opponent_players=[
                PlayerConfig(
                    unum=1, goalie=True,
                    policy_kind="bot",
                    policy_image="HELIOS/helios-base",
                    init_state=PlayerInitState(pos_x=0.9, pos_y=0.5, stamina=6000),
                    blocklist={"dash": True, "catch": False},
                ),
                PlayerConfig(
                    unum=2, goalie=False,
                    policy_kind="bot",
                    policy_image="HELIOS/helios-base",
                    init_state=PlayerInitState(pos_x=0.7, pos_y=0.5),
                ),
            ],
            ally_players=[
                PlayerConfig(
                    unum=1, goalie=True,
                    policy_kind="agent", policy_agent="ssp",
                    policy_image="Cyrus2D/SoccerSimulationProxy",
                    grpc_host="127.0.0.1", grpc_port=6657,
                    init_state=PlayerInitState(pos_x=0.9, pos_y=0.5, stamina=6000),
                    blocklist={"dash": True, "catch": False},
                ),
                PlayerConfig(
                    unum=2, goalie=False,
                    policy_kind="agent", policy_agent="ssp",
                    policy_image="Cyrus2D/SoccerSimulationProxy",
                    grpc_host="127.0.0.1", grpc_port=6657,
                    init_state=PlayerInitState(pos_x=0.7, pos_y=0.5),
                ),
                PlayerConfig(
                    unum=3, goalie=False,
                    policy_kind="agent", policy_agent="ssp",
                    policy_image="Cyrus2D/SoccerSimulationProxy",
                    grpc_host="127.0.0.1", grpc_port=6657,
                    init_state=PlayerInitState(pos_x=0.5, pos_y=0.5),
                ),
            ],
        )
        d = req.to_dict()

        # Top-level structure
        assert d["api_version"] == 1
        assert d["referee"]["enable"] is False
        assert d["stopping"]["time_up"] == 6000
        assert d["stopping"]["goal_l"] == 0
        assert d["init_state"]["ball"]["x"] == 0.5

        # All agents share the same gRPC address
        for p in d["teams"]["allies"]["players"]:
            if p["policy"]["kind"] == "agent":
                assert p["policy"]["grpc_host"] == "127.0.0.1"
                assert p["policy"]["grpc_port"] == 6657
                assert p["policy"]["agent"] == "ssp"
