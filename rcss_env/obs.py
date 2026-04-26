"""Observation vector extraction.

Converts a protobuf WorldModel into a normalised fixed-length float feature
vector (144 dimensions):
  - Self info  (20 dims): position/velocity/speed, stamina, orientations,
    ball relation, reachability, kick/tackle/catch state, and team side
  - Ball info  (14 dims): position/velocity/speed, relative position,
    self-relative distance/angle, and freshness counters
  - Teammates (11 x 5 = 55 dims): per unum 1-11 — visible(1), relative_pos(2), velocity(2)
  - Opponents (11 x 5 = 55 dims): same layout as teammates
"""

from typing import Any

import numpy as np

from .grpc_srv.proto import pb2

# Pitch and physics constants used for normalisation
PITCH_HALF_LENGTH = 52.5
PITCH_HALF_WIDTH = 34.0
MAX_PLAYER_SPEED = 1.5
MAX_BALL_SPEED = 3.0
MAX_STAMINA = 8000.0
PITCH_DIAGONAL = float(np.hypot(PITCH_HALF_LENGTH * 2, PITCH_HALF_WIDTH * 2))
MAX_REACH_STEPS = 20.0


def _clip_ratio(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _clip_positive(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _freshness(count: int) -> float:
    return float(1.0 / (1.0 + max(0, count)))


def _side_feature(side: int) -> float:
    if side == pb2.LEFT:
        return -1.0
    if side == pb2.RIGHT:
        return 1.0
    return 0.0


def dim() -> int:
    """Return the observation vector dimensionality (144)."""
    return 144


def extract(wm: pb2.WorldModel) -> np.ndarray:
    """Extract a normalized feature vector from a WorldModel.

    Args:
        wm: WorldModel protobuf message from a gRPC State.

    Returns:
        A float32 numpy array of shape (144,).
    """
    features: list[float] = []

    # ---- Self info (20 dims) ----
    s = wm.self
    self_speed = float(np.hypot(s.velocity.x, s.velocity.y))
    features.append(_clip_ratio(s.position.x, PITCH_HALF_LENGTH))
    features.append(_clip_ratio(s.position.y, PITCH_HALF_WIDTH))
    features.append(_clip_ratio(s.velocity.x, MAX_PLAYER_SPEED))
    features.append(_clip_ratio(s.velocity.y, MAX_PLAYER_SPEED))
    features.append(_clip_positive(self_speed, MAX_PLAYER_SPEED))
    features.append(_clip_positive(s.stamina, MAX_STAMINA))
    features.append(_clip_ratio(s.body_direction, 180.0))
    features.append(_clip_ratio(s.face_direction, 180.0))
    features.append(_clip_ratio(s.relative_neck_direction, 180.0))
    features.append(_clip_positive(s.dist_from_ball, PITCH_DIAGONAL))
    features.append(_clip_ratio(s.angle_from_ball, 180.0))
    features.append(_clip_positive(s.ball_reach_steps, MAX_REACH_STEPS))
    features.append(float(np.clip(s.kick_rate, 0.0, 1.0)))
    features.append(float(np.clip(s.catch_probability, 0.0, 1.0)))
    features.append(float(np.clip(s.tackle_probability, 0.0, 1.0)))
    features.append(1.0 if s.is_kickable else 0.0)
    features.append(1.0 if s.is_kicking else 0.0)
    features.append(1.0 if s.is_tackling else 0.0)
    features.append(1.0 if s.is_goalie else 0.0)
    features.append(_side_feature(s.side))

    # ---- Ball info (14 dims) ----
    b = wm.ball
    ball_speed = float(np.hypot(b.velocity.x, b.velocity.y))
    features.append(_clip_ratio(b.position.x, PITCH_HALF_LENGTH))
    features.append(_clip_ratio(b.position.y, PITCH_HALF_WIDTH))
    features.append(_clip_ratio(b.velocity.x, MAX_BALL_SPEED))
    features.append(_clip_ratio(b.velocity.y, MAX_BALL_SPEED))
    features.append(_clip_positive(ball_speed, MAX_BALL_SPEED))
    features.append(_clip_ratio(b.relative_position.x, PITCH_HALF_LENGTH * 2))
    features.append(_clip_ratio(b.relative_position.y, PITCH_HALF_WIDTH * 2))
    features.append(_clip_positive(b.dist_from_self, PITCH_DIAGONAL))
    features.append(_clip_ratio(b.angle_from_self, 180.0))
    features.append(_freshness(b.pos_count))
    features.append(_freshness(b.vel_count))
    features.append(_freshness(b.seen_pos_count))
    features.append(_freshness(b.seen_vel_count))
    features.append(_freshness(b.lost_count))

    def _player_features(player_dict: Any, unum: int) -> list[float]:
        """Extract 5-dim features for a single player; returns zeros if not visible."""
        if unum in player_dict:
            p = player_dict[unum]
            return [
                1.0,
                (p.position.x - s.position.x) / (PITCH_HALF_LENGTH * 2),
                (p.position.y - s.position.y) / (PITCH_HALF_WIDTH * 2),
                p.velocity.x / MAX_PLAYER_SPEED,
                p.velocity.y / MAX_PLAYER_SPEED,
            ]
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # ---- Teammates 1-11 (55 dims) ----
    for i in range(1, 12):
        features.extend(_player_features(wm.our_players_dict, i))

    # ---- Opponents 1-11 (55 dims) ----
    for i in range(1, 12):
        features.extend(_player_features(wm.their_players_dict, i))

    return np.array(features, dtype=np.float32)
