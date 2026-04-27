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
    ball_vx: float = 0.0,
    ball_vy: float = 0.0,
    self_x: float = 0.0,
    self_y: float = 0.0,
    self_kickable: bool = False,
    our_side: int = pb2.LEFT,
    cycle: int = 0,
    agent_unum: int = 1,
) -> pb2.WorldModel:
    return pb2.WorldModel(
        our_team_score=our_score,
        their_team_score=their_score,
        our_side=our_side,
        cycle=cycle,
        ball=pb2.Ball(
            position=pb2.Vector2D(x=ball_x, y=ball_y),
            velocity=pb2.Vector2D(x=ball_vx, y=ball_vy),
        ),
        self=pb2.Self(
            position=pb2.Vector2D(x=self_x, y=self_y),
            is_kickable=self_kickable,
        ),
        our_players_dict={
            agent_unum: pb2.Player(
                uniform_number=agent_unum,
                position=pb2.Vector2D(x=self_x, y=self_y),
                dist_from_ball=((self_x - ball_x) ** 2 + (self_y - ball_y) ** 2) ** 0.5,
                ball_reach_steps=0 if self_kickable else 1,
            ),
        },
        kickable_teammate_id=agent_unum if self_kickable else 0,
    )


# Convenience: zero out every shaping/dense term we don't care about.
_OFF = dict(
    reward_agent_to_ball_shaping=0.0,
    reward_ball_to_goal_shaping=0.0,
    reward_ball_velocity_to_goal=0.0,
    reward_kickable_bonus=0.0,
    reward_time_decay=0.0,
    reward_out_of_bounds=0.0,
)


def test_returns_zero_without_previous_observation() -> None:
    reward = _reward()
    assert reward.compute(None, None, _world_model(), _world_model()) == 0.0


def test_uses_goal_difference_delta() -> None:
    reward = _reward(**_OFF)
    prev = _world_model(our_score=1, their_score=1)
    our_goal = _world_model(our_score=2, their_score=1)
    their_goal = _world_model(our_score=1, their_score=2)

    assert reward.compute(prev, prev, our_goal, our_goal) == 10.0
    assert reward.compute(prev, prev, their_goal, their_goal) == -10.0


def test_prefers_truth_over_partial_observation() -> None:
    reward = _reward(**_OFF)
    noisy_prev_obs = _world_model(our_score=0, ball_x=0.0)
    noisy_curr_obs = _world_model(our_score=0, ball_x=-20.0)
    prev_truth = _world_model(our_score=0, ball_x=0.0)
    curr_truth = _world_model(our_score=1, ball_x=20.0)

    assert reward.compute(noisy_prev_obs, prev_truth, noisy_curr_obs, curr_truth) == 10.0


def test_prefers_truth_for_agent_to_ball_shaping() -> None:
    reward = _reward(**{**_OFF, "reward_agent_to_ball_shaping": 1.0})
    noisy_prev_obs = _world_model(self_x=10.0, ball_x=20.0, cycle=1)
    noisy_curr_obs = _world_model(self_x=0.0, ball_x=20.0, cycle=2)
    prev_truth = _world_model(self_x=0.0, ball_x=20.0, cycle=1)
    curr_truth = _world_model(self_x=10.0, ball_x=20.0, cycle=2)

    assert reward.compute(noisy_prev_obs, prev_truth, noisy_curr_obs, curr_truth) > 0.0


def test_prefers_truth_for_kickable_bonus() -> None:
    reward = _reward(**{**_OFF, "reward_kickable_bonus": 0.5})
    noisy_prev_obs = _world_model(self_kickable=True, cycle=1)
    noisy_curr_obs = _world_model(self_kickable=False, cycle=2)
    prev_truth = _world_model(self_kickable=False, cycle=1)
    curr_truth = _world_model(self_kickable=True, cycle=2)

    assert reward.compute(noisy_prev_obs, prev_truth, noisy_curr_obs, curr_truth) == pytest.approx(0.5)


# ---- Agent -> ball shaping (the critical "approach the ball" signal) ------

def test_agent_to_ball_shaping_rewards_closing_in() -> None:
    overrides = {**_OFF, "reward_agent_to_ball_shaping": 1.0}
    reward = _reward(**overrides)
    prev = _world_model(self_x=0.0, self_y=0.0, ball_x=20.0, ball_y=0.0, cycle=1)
    closer = _world_model(self_x=10.0, self_y=0.0, ball_x=20.0, ball_y=0.0, cycle=2)
    farther = _world_model(self_x=-10.0, self_y=0.0, ball_x=20.0, ball_y=0.0, cycle=2)

    r_closer = reward.compute(prev, prev, closer, closer)
    r_farther = reward.compute(prev, prev, farther, farther)

    assert r_closer > 0.0
    assert r_farther < 0.0
    assert r_closer > r_farther


# ---- Ball -> goal shaping --------------------------------------------------

def test_ball_to_goal_shaping_rewards_progress() -> None:
    overrides = {**_OFF, "reward_ball_to_goal_shaping": 1.0}
    reward = _reward(**overrides)
    prev = _world_model(ball_x=0.0, cycle=1)
    closer = _world_model(ball_x=10.0, cycle=2)
    farther = _world_model(ball_x=-10.0, cycle=2)

    assert reward.compute(prev, prev, closer, closer) > 0.0
    assert reward.compute(prev, prev, farther, farther) < 0.0


def test_ball_to_goal_shaping_respects_right_side() -> None:
    overrides = {**_OFF, "reward_ball_to_goal_shaping": 1.0, "team_side": "right"}
    reward = _reward(**overrides)
    prev = _world_model(ball_x=0.0, our_side=pb2.RIGHT, cycle=1)
    closer = _world_model(ball_x=-10.0, our_side=pb2.RIGHT, cycle=2)
    farther = _world_model(ball_x=10.0, our_side=pb2.RIGHT, cycle=2)

    assert reward.compute(prev, prev, closer, closer) > 0.0
    assert reward.compute(prev, prev, farther, farther) < 0.0


# ---- Out-of-bounds is edge-triggered ---------------------------------------

def test_out_of_bounds_only_on_transition() -> None:
    overrides = {**_OFF, "reward_out_of_bounds": 2.0}
    reward = _reward(**overrides)
    in_bounds = _world_model(ball_x=0.0, ball_y=0.0)
    just_out = _world_model(ball_x=0.0, ball_y=35.0)
    still_out = _world_model(ball_x=0.0, ball_y=36.0)

    # Transition in-bounds -> out: penalised once.
    assert reward.compute(in_bounds, in_bounds, just_out, just_out) == -2.0
    # Already out and remains out: NOT penalised again (was a bug previously).
    assert reward.compute(just_out, just_out, still_out, still_out) == 0.0


def test_goal_mouth_crossing_not_treated_as_out_of_bounds() -> None:
    overrides = {**_OFF, "reward_out_of_bounds": 2.0}
    reward = _reward(**overrides)
    prev = _world_model(ball_x=52.0, ball_y=0.0)
    curr = _world_model(ball_x=53.0, ball_y=0.0)
    assert reward.compute(prev, prev, curr, curr) == 0.0


# ---- Kickable rising-edge bonus -------------------------------------------

def test_kickable_rising_edge_bonus() -> None:
    overrides = {**_OFF, "reward_kickable_bonus": 0.5}
    reward = _reward(**overrides)
    not_kickable = _world_model(self_kickable=False, cycle=1)
    kickable = _world_model(self_kickable=True, cycle=2)

    # Rising edge: bonus.
    assert reward.compute(not_kickable, not_kickable, kickable, kickable) == pytest.approx(0.5)
    # Held high: no double bonus.
    assert reward.compute(kickable, kickable, kickable, kickable) == 0.0


# ---- Ball velocity toward goal ---------------------------------------------

def test_ball_velocity_toward_goal_rewards_forward_kick() -> None:
    overrides = {**_OFF, "reward_ball_velocity_to_goal": 1.0}
    reward = _reward(**overrides)
    prev = _world_model(ball_x=0.0, ball_vx=0.0, cycle=1)
    forward = _world_model(ball_x=0.0, ball_vx=2.0, cycle=2)
    backward = _world_model(ball_x=0.0, ball_vx=-2.0, cycle=2)

    assert reward.compute(prev, prev, forward, forward) > 0.0
    assert reward.compute(prev, prev, backward, backward) < 0.0


# ---- Time decay ------------------------------------------------------------

def test_time_decay_only_when_not_scoring() -> None:
    overrides = {**_OFF, "reward_time_decay": 0.01}
    reward = _reward(**overrides)
    prev = _world_model(cycle=10)
    curr_no_goal = _world_model(cycle=13)
    curr_goal = _world_model(our_score=1, cycle=13)

    assert reward.compute(prev, prev, curr_no_goal, curr_no_goal) == pytest.approx(-0.03)
    # No time decay on the scoring step.
    assert reward.compute(prev, prev, curr_goal, curr_goal) == pytest.approx(10.0)


def test_time_decay_capped_for_huge_cycle_gap() -> None:
    overrides = {**_OFF, "reward_time_decay": 0.01, "max_cycle_gap": 5}
    reward = _reward(**overrides)
    prev = _world_model(cycle=10)
    curr = _world_model(cycle=10 + 100)  # massive gap from a stoppage
    # Capped at 5 cycles -> -0.05 instead of -1.00.
    assert reward.compute(prev, prev, curr, curr) == pytest.approx(-0.05)


# ---- Breakdown logging hook -----------------------------------------------

def test_last_breakdown_is_populated() -> None:
    reward = _reward()
    prev = _world_model(self_x=0.0, ball_x=20.0, cycle=1)
    curr = _world_model(self_x=10.0, ball_x=20.0, cycle=2)
    total = reward.compute(prev, prev, curr, curr)

    bd = reward.last_breakdown
    assert bd.agent_to_ball_shaping > 0.0
    assert bd.total() == pytest.approx(total)


def test_reward_fn_mixin_breakdown_dict_is_populated() -> None:
    reward = _reward()
    prev = _world_model(self_x=0.0, ball_x=20.0, cycle=1)
    curr = _world_model(self_x=10.0, ball_x=20.0, cycle=2)

    total = reward.compute(prev, prev, curr, curr)

    breakdown = reward.last_reward_breakdown
    assert breakdown["agent_to_ball_shaping"] > 0.0
    assert sum(breakdown.values()) == pytest.approx(total)
