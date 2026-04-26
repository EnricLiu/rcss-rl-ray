from __future__ import annotations

import pytest

from rcss_env.grpc_srv.proto import pb2
from train.curriculum.shooting import ShootingCurriculumConfig, ShootingReward


def _reward(**kwargs: object) -> ShootingReward:
    return ShootingReward(
        ShootingCurriculumConfig(
            agent_unum=1,
            our_player_num=1,
            oppo_player_num=1,
            our_goalie_unum=None,
            oppo_goalie_unum=None,
            **kwargs,
        )
    )


def _world_model(
    our_score: int = 0,
    their_score: int = 0,
    ball_x: float = 0.0,
    ball_y: float = 0.0,
    our_side: int = pb2.LEFT,
    cycle: int = 0,
) -> pb2.WorldModel:
    return pb2.WorldModel(
        our_team_score=our_score,
        their_team_score=their_score,
        our_side=our_side,
        cycle=cycle,
        ball=pb2.Ball(position=pb2.Vector2D(x=ball_x, y=ball_y)),
    )


def test_shooting_reward_returns_zero_without_previous_observation() -> None:
    reward = _reward()

    assert reward.compute(None, None, _world_model(), _world_model()) == 0.0


def test_shooting_reward_uses_goal_difference_delta() -> None:
    reward = _reward(reward_ball_to_goal_shaping=0.0, reward_time_decay=0.0)
    prev = _world_model(our_score=1, their_score=1)
    our_goal = _world_model(our_score=2, their_score=1)
    their_goal = _world_model(our_score=1, their_score=2)
    both_goal = _world_model(our_score=2, their_score=2)

    assert reward.compute(prev, prev, our_goal, our_goal) == 10.0
    assert reward.compute(prev, prev, their_goal, their_goal) == -10.0
    assert reward.compute(prev, prev, both_goal, both_goal) == 0.0


def test_shooting_reward_prefers_truth_over_partial_observation() -> None:
    reward = _reward(reward_ball_to_goal_shaping=0.0, reward_time_decay=0.0)
    noisy_prev_obs = _world_model(our_score=0, their_score=0, ball_x=0.0)
    noisy_curr_obs = _world_model(our_score=0, their_score=0, ball_x=-20.0)
    prev_truth = _world_model(our_score=0, their_score=0, ball_x=0.0)
    curr_truth = _world_model(our_score=1, their_score=0, ball_x=20.0)

    assert reward.compute(noisy_prev_obs, prev_truth, noisy_curr_obs, curr_truth) == 10.0


def test_shooting_reward_shapes_with_truth_ball_position_not_noisy_obs() -> None:
    reward = _reward(reward_time_decay=0.0)
    noisy_prev_obs = _world_model(ball_x=0.0, cycle=1)
    noisy_curr_obs = _world_model(ball_x=-10.0, cycle=2)
    prev_truth = _world_model(ball_x=0.0, cycle=1)
    curr_truth = _world_model(ball_x=10.0, cycle=2)

    assert reward.compute(noisy_prev_obs, prev_truth, noisy_curr_obs, curr_truth) > 0.0


def test_shooting_reward_shapes_progress_toward_opponent_goal() -> None:
    reward = _reward(reward_time_decay=0.0)
    prev = _world_model(ball_x=0.0, ball_y=0.0, cycle=1)

    closer = reward.compute(
        prev,
        prev,
        _world_model(ball_x=10.0, ball_y=0.0, cycle=2),
        _world_model(ball_x=10.0, ball_y=0.0),
    )
    farther = reward.compute(
        prev,
        prev,
        _world_model(ball_x=-10.0, ball_y=0.0, cycle=2),
        _world_model(ball_x=-10.0, ball_y=0.0),
    )

    assert closer > 0.0
    assert farther < 0.0
    assert closer == pytest.approx(-farther)


def test_shooting_reward_respects_right_side_attack_direction() -> None:
    reward = _reward(team_side="right", reward_time_decay=0.0)
    prev = _world_model(ball_x=0.0, ball_y=0.0, our_side=pb2.RIGHT, cycle=1)

    closer = reward.compute(
        prev,
        prev,
        _world_model(ball_x=-10.0, ball_y=0.0, our_side=pb2.RIGHT, cycle=2),
        _world_model(ball_x=-10.0, ball_y=0.0, our_side=pb2.RIGHT),
    )
    farther = reward.compute(
        prev,
        prev,
        _world_model(ball_x=10.0, ball_y=0.0, our_side=pb2.RIGHT, cycle=2),
        _world_model(ball_x=10.0, ball_y=0.0, our_side=pb2.RIGHT),
    )

    assert closer > 0.0
    assert farther < 0.0


def test_shooting_reward_penalizes_non_goal_out_of_bounds() -> None:
    reward = _reward(
        reward_out_of_bounds=2.0,
        reward_ball_to_goal_shaping=0.0,
        reward_time_decay=0.0,
    )
    prev = _world_model(ball_x=0.0, ball_y=0.0)
    curr = _world_model(ball_x=0.0, ball_y=35.0)

    assert reward.compute(prev, prev, curr, curr) == -2.0


def test_shooting_reward_does_not_penalize_goal_mouth_crossing_as_out_of_bounds() -> None:
    reward = _reward(
        reward_out_of_bounds=2.0,
        reward_ball_to_goal_shaping=0.0,
        reward_time_decay=0.0,
    )
    prev = _world_model(ball_x=52.0, ball_y=0.0)
    curr = _world_model(ball_x=53.0, ball_y=0.0)

    assert reward.compute(prev, prev, curr, curr) == 0.0


def test_shooting_reward_applies_time_decay_by_cycle_delta() -> None:
    reward = _reward(
        reward_ball_to_goal_shaping=0.0,
        reward_time_decay=0.01,
    )
    prev = _world_model(cycle=10)
    curr = _world_model(cycle=13)

    assert reward.compute(prev, prev, curr, curr) == pytest.approx(-0.03)
