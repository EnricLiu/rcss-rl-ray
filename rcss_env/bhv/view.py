#! https://github.com/helios-base/helios-base/blob/master/src/player/view_tactical.cpp
import math
from typing import override, Optional

from ..grpc_srv.proto import pb2
from .bhv import BhvMixin

__ALL__ = ["BhvHeliosView"]

DEFAULT_PITCH_HALF_LENGTH = 52.5
DEFAULT_OUR_PENALTY_AREA_LINE_X = -36.0
DEFAULT_MAX_NECK_ANGLE = 90.0
VIEW_NORMAL_HALF_WIDTH_MINUS_BUF = 57.0  # 120 / 2 - 3
VIEW_NARROW_HALF_WIDTH_MINUS_BUF = 27.0  # 60 / 2 - 3


def predicted_ball_position(wm: pb2.WorldModel) -> tuple[float, float]:
    ball = wm.ball
    self_player = wm.self
    intercept = wm.intercept_table

    reach_steps = [
        intercept.self_reach_steps,
        intercept.first_teammate_reach_steps,
        intercept.first_opponent_reach_steps,
    ]
    valid_steps = [step for step in reach_steps if step >= 0]
    ball_reach_cycle = min(valid_steps, default=0)

    ball_x = ball.position.x + ball.velocity.x * ball_reach_cycle
    ball_y = ball.position.y + ball.velocity.y * ball_reach_cycle

    # Fall back to relative coordinates if absolute position looks unavailable.
    if (
        ball_x == 0.0
        and ball_y == 0.0
        and ball.relative_position.x == 0.0
        and ball.relative_position.y == 0.0
        and ball.dist_from_self > 0.0
    ):
        ball_x = self_player.position.x + ball.relative_position.x
        ball_y = self_player.position.y + ball.relative_position.y

    return ball_x, ball_y


def ball_position_valid(ball: pb2.Ball) -> bool:
    return any(
        (
            ball.pos_count,
            ball.seen_pos_count,
            ball.heard_pos_count,
            ball.position.x,
            ball.position.y,
            ball.relative_position.x,
            ball.relative_position.y,
            ball.dist_from_self,
        )
    )


def ball_dist(wm: pb2.WorldModel, ball_x: float, ball_y: float) -> float:
    self_pos = wm.self.position
    dist = math.hypot(self_pos.x - ball_x, self_pos.y - ball_y)
    if dist < 1e-3 and wm.ball.dist_from_self > 0.0:
        return wm.ball.dist_from_self
    return dist


def nearest_ball_dist(players: list[pb2.Player]) -> float:
    if not players:
        return 1000.0
    return min(player.dist_from_ball for player in players)


class BhvHeliosView(BhvMixin):
    @staticmethod
    def __map_action(action_name: str) -> pb2.PlayerAction:
        action_map = {
            "view_synch": pb2.View_Synch(),
            "view_normal": pb2.View_Normal(),
            "view_wide": pb2.View_Wide(),
        }
        return pb2.PlayerAction(**{action_name: action_map[action_name]})

    @staticmethod
    def __do_our_goalie_free_kick(wm: pb2.WorldModel) -> pb2.PlayerAction:
        if wm.self.is_goalie:
            return BhvHeliosView.__map_action("view_wide")
        return BhvHeliosView.__do_default(wm)

    @staticmethod
    def __do_default(wm: pb2.WorldModel) -> pb2.PlayerAction:
        if not ball_position_valid(wm.ball):
            return BhvHeliosView.__map_action("view_wide")

        ball_x, ball_y = predicted_ball_position(wm)
        dist_to_ball = ball_dist(wm, ball_x, ball_y)
        ball_angle = abs(wm.ball.angle_from_self)

        if wm.self.is_goalie and not wm.self.is_kickable:
            goal_x = -DEFAULT_PITCH_HALF_LENGTH
            goal_y = 0.0
            if (
                dist_to_ball > 10.0
                or ball_x > DEFAULT_OUR_PENALTY_AREA_LINE_X
                or math.hypot(ball_x - goal_x, ball_y - goal_y) > 20.0
            ):
                angle_diff = ball_angle - DEFAULT_MAX_NECK_ANGLE
                if angle_diff > VIEW_NORMAL_HALF_WIDTH_MINUS_BUF:
                    return BhvHeliosView.__map_action("view_wide")
                if angle_diff > VIEW_NARROW_HALF_WIDTH_MINUS_BUF:
                    return BhvHeliosView.__map_action("view_normal")

        if dist_to_ball > 40.0:
            return BhvHeliosView.__map_action("view_wide")

        if dist_to_ball > 20.0:
            return BhvHeliosView.__map_action("view_normal")

        if dist_to_ball > 10.0 and ball_angle > 120.0:
            return BhvHeliosView.__map_action("view_wide")

        teammate_ball_dist = nearest_ball_dist(list(wm.teammates))
        opponent_ball_dist = nearest_ball_dist(list(wm.opponents))

        if (
            not wm.self.is_goalie
            and teammate_ball_dist > 5.0
            and opponent_ball_dist > 5.0
            and dist_to_ball > 10.0
            and wm.ball.dist_from_self > 10.0
        ):
            return BhvHeliosView.__map_action("view_normal")

        return BhvHeliosView.__map_action("view_synch")

    @override
    def parse(self, wm: Optional[pb2.WorldModel]) -> pb2.PlayerAction:
        if wm is None: return BhvHeliosView.__map_action("view_wide")

        game_mode = wm.game_mode_type

        if game_mode in {
            pb2.BeforeKickOff,
            pb2.AfterGoal_,
            pb2.PenaltySetup_,
            pb2.PenaltyReady_,
            pb2.PenaltyMiss_,
            pb2.PenaltyScore_,
        }:
            return BhvHeliosView.__map_action("view_wide")

        if game_mode == pb2.PenaltyTaken_:
            return BhvHeliosView.__map_action("view_synch")

        if game_mode == pb2.GoalieCatch_ and wm.is_our_set_play:
            return BhvHeliosView.__do_our_goalie_free_kick(wm)

        return BhvHeliosView.__do_default(wm)
