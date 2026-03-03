"""Observation vector extraction.

Converts a protobuf WorldModel into a normalised fixed-length float feature
vector (124 dimensions):
  - Self info  (8 dims): position(2), velocity(2), stamina(1), body_dir(1), neck_rel_dir(1), kickable(1)
  - Ball info  (6 dims): position(2), velocity(2), relative_position(2)
  - Teammates (11 x 5 = 55 dims): per unum 1-11 — visible(1), relative_pos(2), velocity(2)
  - Opponents (11 x 5 = 55 dims): same layout as teammates
"""

from typing import Any

import numpy as np

from .grpc_srv import pb2

# Pitch and physics constants used for normalisation
PITCH_HALF_LENGTH = 52.5
PITCH_HALF_WIDTH = 34.0
MAX_PLAYER_SPEED = 1.5
MAX_BALL_SPEED = 3.0
MAX_STAMINA = 8000.0


def dim() -> int:
    """Return the observation vector dimensionality (124)."""
    return 124


def extract(wm: pb2.WorldModel) -> np.ndarray:
    """Extract a normalised feature vector from a WorldModel.

    Args:
        wm: WorldModel protobuf message from a gRPC State.

    Returns:
        A float32 numpy array of shape (124,).
    """
    features: list[float] = []

    # ---- Self info (8 dims) ----
    s = wm.self
    features.append(s.position.x / PITCH_HALF_LENGTH)
    features.append(s.position.y / PITCH_HALF_WIDTH)
    features.append(s.velocity.x / MAX_PLAYER_SPEED)
    features.append(s.velocity.y / MAX_PLAYER_SPEED)
    features.append(s.stamina / MAX_STAMINA)
    features.append(s.body_direction / 180.0)
    features.append(s.relative_neck_direction / 180.0)
    features.append(1.0 if s.is_kickable else 0.0)

    # ---- Ball info (6 dims) ----
    b = wm.ball
    features.append(b.position.x / PITCH_HALF_LENGTH)
    features.append(b.position.y / PITCH_HALF_WIDTH)
    features.append(b.velocity.x / MAX_BALL_SPEED)
    features.append(b.velocity.y / MAX_BALL_SPEED)
    features.append((b.position.x - s.position.x) / (PITCH_HALF_LENGTH * 2))
    features.append((b.position.y - s.position.y) / (PITCH_HALF_WIDTH * 2))

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
