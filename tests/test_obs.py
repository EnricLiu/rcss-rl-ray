from __future__ import annotations

import pytest

from rcss_env import obs as observation
from rcss_env.grpc_srv.proto import pb2


def _world_model() -> pb2.WorldModel:
    return pb2.WorldModel(
        self=pb2.Self(
            position=pb2.Vector2D(x=10.0, y=-5.0),
            velocity=pb2.Vector2D(x=1.2, y=-0.5),
            side=pb2.RIGHT,
            is_goalie=False,
            body_direction=90.0,
            face_direction=45.0,
            is_kicking=True,
            dist_from_ball=12.0,
            angle_from_ball=-30.0,
            ball_reach_steps=4,
            is_tackling=False,
            relative_neck_direction=-15.0,
            stamina=6000.0,
            is_kickable=True,
            catch_probability=0.2,
            tackle_probability=0.3,
            kick_rate=0.8,
        ),
        ball=pb2.Ball(
            position=pb2.Vector2D(x=20.0, y=3.0),
            relative_position=pb2.Vector2D(x=10.0, y=8.0),
            velocity=pb2.Vector2D(x=2.0, y=-1.0),
            pos_count=1,
            seen_pos_count=2,
            vel_count=3,
            seen_vel_count=4,
            lost_count=5,
            dist_from_self=12.0,
            angle_from_self=25.0,
        ),
        our_players_dict={
            1: pb2.Player(
                position=pb2.Vector2D(x=12.0, y=-4.0),
                velocity=pb2.Vector2D(x=0.1, y=0.2),
            )
        },
        their_players_dict={
            7: pb2.Player(
                position=pb2.Vector2D(x=18.0, y=2.0),
                velocity=pb2.Vector2D(x=-0.2, y=0.3),
            )
        },
    )


def test_observation_dim_matches_expanded_layout() -> None:
    wm = _world_model()

    vector = observation.extract(wm)

    assert observation.dim() == 144
    assert vector.shape == (144,)
    assert vector.dtype.name == "float32"


def test_observation_includes_expanded_self_and_ball_features() -> None:
    vector = observation.extract(_world_model())

    assert vector[0] == pytest.approx(10.0 / observation.PITCH_HALF_LENGTH)
    assert vector[4] == pytest.approx(((1.2**2 + (-0.5) ** 2) ** 0.5) / observation.MAX_PLAYER_SPEED)
    assert vector[12] == pytest.approx(0.8)
    assert vector[15] == pytest.approx(1.0)  # is_kickable
    assert vector[17] == pytest.approx(0.0)  # is_tackling
    assert vector[19] == pytest.approx(1.0)  # right side

    assert vector[20] == pytest.approx(20.0 / observation.PITCH_HALF_LENGTH)
    assert vector[24] == pytest.approx(((2.0**2 + (-1.0) ** 2) ** 0.5) / observation.MAX_BALL_SPEED)
    assert vector[27] == pytest.approx(12.0 / observation.PITCH_DIAGONAL)
    assert vector[29] == pytest.approx(1.0 / 2.0)  # pos freshness for pos_count=1
    assert vector[33] == pytest.approx(1.0 / 6.0)  # lost freshness for lost_count=5
