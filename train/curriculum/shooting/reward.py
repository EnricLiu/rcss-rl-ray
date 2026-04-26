"""Reward function for the shooting curriculum (single-agent).

Design goals (in order of priority):
  1. Make "approach the ball" learnable: dense agent->ball potential-based
     shaping + one-shot bonus on first kickable contact.
  2. Make "kick the ball toward the goal" learnable: ball->goal potential-based
     shaping + dense ball-velocity-to-goal term.
  3. Sparse, high-magnitude terminal-style rewards on goals / concedes.
  4. Edge-triggered penalties for ball going out of bounds (not in goal mouth).
  5. Robustness: gamma-aware PBRS preserves the optimal policy, all shaping
     terms are clipped to suppress spikes from referee resets / teleports.
"""

import math
from dataclasses import dataclass, asdict
from typing import override

from rcss_env.grpc_srv.proto import pb2
from rcss_env.reward import RewardFnMixin

from .config import ShootingCurriculumConfig


_PITCH_HALF_LENGTH = 52.5
_PITCH_HALF_WIDTH = 34.0
_PITCH_LENGTH = _PITCH_HALF_LENGTH * 2
_GOAL_HALF_WIDTH = 3.66
_PITCH_DIAGONAL = math.hypot(_PITCH_LENGTH, _PITCH_HALF_WIDTH * 2)
_MAX_BALL_TO_GOAL_DISTANCE = math.hypot(_PITCH_LENGTH, _PITCH_HALF_WIDTH)
_MAX_BALL_SPEED = 3.0  # rcssserver default ball_speed_max


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _target_goal_x(world_model: pb2.WorldModel, fallback_side: str) -> float:
    match world_model.our_side:
        case pb2.LEFT:
            return _PITCH_HALF_LENGTH
        case pb2.RIGHT:
            return -_PITCH_HALF_LENGTH
        case _:
            return -_PITCH_HALF_LENGTH if fallback_side == "right" else _PITCH_HALF_LENGTH


def _ball_to_goal_distance(wm: pb2.WorldModel, fallback_side: str) -> float:
    target_x = _target_goal_x(wm, fallback_side)
    return math.hypot(target_x - wm.ball.position.x, wm.ball.position.y)


def _agent_to_ball_distance(wm: pb2.WorldModel) -> float:
    """Prefer the precomputed `self.dist_from_ball`; fall back to recompute."""
    s = wm.self
    d = float(s.dist_from_ball)
    if d > 0.0:
        return d
    return math.hypot(s.position.x - wm.ball.position.x,
                      s.position.y - wm.ball.position.y)


def _ball_velocity_toward_goal(wm: pb2.WorldModel, fallback_side: str) -> float:
    """Component of the ball velocity along (ball -> goal-centre)."""
    target_x = _target_goal_x(wm, fallback_side)
    dx = target_x - wm.ball.position.x
    dy = -wm.ball.position.y  # goal centre y == 0
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    return (wm.ball.velocity.x * dx + wm.ball.velocity.y * dy) / norm


def _is_ball_out_of_bounds(wm: pb2.WorldModel) -> bool:
    p = wm.ball.position
    return abs(p.x) > _PITCH_HALF_LENGTH or abs(p.y) > _PITCH_HALF_WIDTH


def _is_ball_in_goal_mouth(wm: pb2.WorldModel) -> bool:
    p = wm.ball.position
    return abs(p.x) > _PITCH_HALF_LENGTH and abs(p.y) <= _GOAL_HALF_WIDTH


def _score_delta(prev: pb2.WorldModel, curr: pb2.WorldModel) -> tuple[int, int]:
    return (
        curr.our_team_score - prev.our_team_score,
        curr.their_team_score - prev.their_team_score,
    )


def _clip(x: float, limit: float) -> float:
    if x > limit:
        return limit
    if x < -limit:
        return -limit
    return x


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Per-step breakdown of all reward components (for logging / debugging)."""
    goal: float = 0.0
    concede: float = 0.0
    out_of_bounds: float = 0.0
    kickable_bonus: float = 0.0
    agent_to_ball_shaping: float = 0.0
    ball_to_goal_shaping: float = 0.0
    ball_velocity_to_goal: float = 0.0
    time_decay: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    def total(self) -> float:
        return float(sum(self.to_dict().values()))


class ShootingReward(RewardFnMixin):
    def __init__(self, config: ShootingCurriculumConfig):
        super().__init__()
        self.config = config
        # Last computed breakdown — exposed for callbacks / tests.
        self.last_breakdown: RewardBreakdown = RewardBreakdown()

    # -- Potential functions -------------------------------------------------
    def _phi_agent_to_ball(self, wm: pb2.WorldModel) -> float:
        # Negative distance, normalized to roughly [-1, 0].
        return -_agent_to_ball_distance(wm) / _PITCH_DIAGONAL

    def _phi_ball_to_goal(self, wm: pb2.WorldModel) -> float:
        # Negative distance, normalized to roughly [-1, 0].
        return -_ball_to_goal_distance(wm, self.config.team_side) / _MAX_BALL_TO_GOAL_DISTANCE

    @override
    def compute(
        self,
        prev_obs: pb2.WorldModel | None,
        _prev_truth: pb2.WorldModel | None,
        curr_obs: pb2.WorldModel,
        _curr_truth: pb2.WorldModel,
    ) -> float:
        # Use full-information world models whenever possible.
        prev_wm = prev_obs
        if prev_wm is None:
            self.last_breakdown = RewardBreakdown()
            self.reset_reward_breakdown()
            return 0.0
        curr_wm = curr_obs

        cfg = self.config
        bd = RewardBreakdown()

        # -- 1) Sparse goal events --------------------------------------------
        our_goal_delta, their_goal_delta = _score_delta(prev_wm, curr_wm)
        scored = (our_goal_delta != 0) or (their_goal_delta != 0)
        bd.goal = cfg.reward_goal * float(our_goal_delta)
        bd.concede = -cfg.reward_concede * float(their_goal_delta)

        # -- 2) Out-of-bounds (edge-triggered, ignore goal-mouth crossings) ---
        prev_oob = _is_ball_out_of_bounds(prev_wm)
        curr_oob = _is_ball_out_of_bounds(curr_wm)
        if (
            not scored
            and curr_oob
            and not prev_oob
            and not _is_ball_in_goal_mouth(curr_wm)
        ):
            bd.out_of_bounds = -cfg.reward_out_of_bounds

        # -- 3) Kickable rising-edge bonus -----------------------------------
        # `self` here refers to the agent because curr_wm is the agent-centric
        # full world model (env passes `state.full_world_model`).
        prev_kickable = bool(prev_wm.self.is_kickable)
        curr_kickable = bool(curr_wm.self.is_kickable)
        if curr_kickable:
            bd.kickable_bonus += cfg.reward_kickable_bonus
            if not prev_kickable:
                bd.kickable_bonus += cfg.reward_kickable_bonus

        # -- Cycle gap (used by shaping & time decay) -------------------------
        raw_gap = curr_wm.cycle - prev_wm.cycle
        # If the gap is non-positive (e.g. wrap-around) or larger than the cap
        # (a stoppage happened between two PlayOn frames), suppress shaping &
        # use a small cap for time decay to avoid spikes.
        skip_shaping = (raw_gap <= 0) or (raw_gap > cfg.max_cycle_gap)
        cycle_gap = max(1, min(raw_gap, cfg.max_cycle_gap))

        # -- 4) Potential-based shaping (gamma-aware) -------------------------
        # F = gamma * Phi(s') - Phi(s) ; preserves optimal policy.
        if not skip_shaping:
            gamma = cfg.gamma_shaping
            clip_lim = cfg.shaping_clip

            phi_a_prev = self._phi_agent_to_ball(prev_wm)
            phi_a_curr = self._phi_agent_to_ball(curr_wm)
            f_agent = _clip(gamma * phi_a_curr - phi_a_prev, clip_lim)
            bd.agent_to_ball_shaping = cfg.reward_agent_to_ball_shaping * f_agent

            phi_b_prev = self._phi_ball_to_goal(prev_wm)
            phi_b_curr = self._phi_ball_to_goal(curr_wm)
            f_ball = _clip(gamma * phi_b_curr - phi_b_prev, clip_lim)
            bd.ball_to_goal_shaping = cfg.reward_ball_to_goal_shaping * f_ball

            # -- 5) Dense ball-velocity-toward-goal --------------------------
            v_to_goal = _ball_velocity_toward_goal(curr_wm, cfg.team_side)
            v_norm = max(-1.0, min(1.0, v_to_goal / _MAX_BALL_SPEED))
            bd.ball_velocity_to_goal = cfg.reward_ball_velocity_to_goal * v_norm

        # -- 6) Time decay (only when not scoring) ----------------------------
        if not scored:
            bd.time_decay = -cfg.reward_time_decay * cycle_gap

        self.last_breakdown = bd
        self.set_reward_breakdown(bd.to_dict())
        return bd.total()
