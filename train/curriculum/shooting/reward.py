import math
from typing import override

from rcss_env.grpc_srv.proto import pb2
from rcss_env.reward import RewardFnMixin

from .config import ShootingCurriculumConfig


_PITCH_HALF_LENGTH = 52.5
_PITCH_HALF_WIDTH = 34.0
_PITCH_LENGTH = _PITCH_HALF_LENGTH * 2
_GOAL_HALF_WIDTH = 3.66
_MAX_BALL_TO_GOAL_DISTANCE = math.hypot(_PITCH_LENGTH, _PITCH_HALF_WIDTH)


def _score_delta(prev: pb2.WorldModel, curr: pb2.WorldModel) -> tuple[int, int]:
    return (
        curr.our_team_score - prev.our_team_score,
        curr.their_team_score - prev.their_team_score,
    )


def _ball_to_goal_distance(ball_wm: pb2.WorldModel, side_wm: pb2.WorldModel, fallback_side: str) -> float:
    target_x = _target_goal_x(side_wm, fallback_side)
    return math.hypot(target_x - ball_wm.ball.position.x, ball_wm.ball.position.y)


def _target_goal_x(world_model: pb2.WorldModel, fallback_side: str) -> float:
    match world_model.our_side:
        case pb2.LEFT:
            return _PITCH_HALF_LENGTH
        case pb2.RIGHT:
            return -_PITCH_HALF_LENGTH
        case _:
            return -_PITCH_HALF_LENGTH if fallback_side == "right" else _PITCH_HALF_LENGTH


def _is_ball_out_of_bounds(world_model: pb2.WorldModel) -> bool:
    ball = world_model.ball.position
    return abs(ball.x) > _PITCH_HALF_LENGTH or abs(ball.y) > _PITCH_HALF_WIDTH


def _is_ball_in_goal_mouth(world_model: pb2.WorldModel) -> bool:
    ball = world_model.ball.position
    return abs(ball.x) > _PITCH_HALF_LENGTH and abs(ball.y) <= _GOAL_HALF_WIDTH


def _elapsed_cycles(prev: pb2.WorldModel, curr: pb2.WorldModel) -> int:
    return max(1, curr.cycle - prev.cycle)


class ShootingReward(RewardFnMixin):
    def __init__(self, config: ShootingCurriculumConfig):
        self.config = config

    @override
    def compute(
        self,
        prev_obs: pb2.WorldModel | None,
        prev_truth: pb2.WorldModel | None,
        curr_obs: pb2.WorldModel,
        curr_truth: pb2.WorldModel
    ) -> float:
        prev_reward_wm = prev_truth if prev_truth is not None else prev_obs
        if prev_reward_wm is None:
            return 0.0

        # Rewards should use full-information world models whenever available;
        # the policy observation can be partial/noisy.
        curr_reward_wm = curr_truth
        our_goal_delta, their_goal_delta = _score_delta(prev_reward_wm, curr_reward_wm)

        reward = 0.0
        reward += self.config.reward_goal * our_goal_delta
        reward -= self.config.reward_concede * their_goal_delta

        scored = our_goal_delta != 0 or their_goal_delta != 0
        if (
            not scored
            and _is_ball_out_of_bounds(curr_reward_wm)
            and not _is_ball_in_goal_mouth(curr_reward_wm)
        ):
            reward -= self.config.reward_out_of_bounds

        prev_distance = _ball_to_goal_distance(prev_reward_wm, prev_reward_wm, self.config.team_side)
        curr_distance = _ball_to_goal_distance(curr_reward_wm, curr_reward_wm, self.config.team_side)
        distance_delta = (prev_distance - curr_distance) / _MAX_BALL_TO_GOAL_DISTANCE
        reward += self.config.reward_ball_to_goal_shaping * distance_delta

        reward -= self.config.reward_time_decay * _elapsed_cycles(prev_reward_wm, curr_reward_wm)

        return float(reward)
