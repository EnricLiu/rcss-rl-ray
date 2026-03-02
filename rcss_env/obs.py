from typing import Any

import numpy as np

from .grpc_srv import pb2

PITCH_HALF_LENGTH = 52.5
PITCH_HALF_WIDTH = 34.0
MAX_PLAYER_SPEED = 1.5
MAX_BALL_SPEED = 3.0
MAX_STAMINA = 8000.0

def dim() -> int:
    """Return the dimensionality of the observation vector."""
    return 124

def extract(wm: pb2.WorldModel) -> np.ndarray:
    """Convert a WorldModel to a (124,) normalised numpy array.

    Structure (124 dims total):
    - Self (8): Pos(2), Vel(2), Stamina(1), BodyDir(1), NeckDir(1), IsKickable(1)
    - Ball (6): Pos(2), Vel(2), RelPos(2)
    - Teammates (11 × 5): [Valid(1), RelPos(2), Vel(2)] × 11 (ordered by unum 1-11)
    - Opponents (11 × 5): [Valid(1), RelPos(2), Vel(2)] × 11 (ordered by unum 1-11)
    """
    features: list[float] = []

    # --- 1. Self Information (8 dims) ---
    s = wm.self
    features.append(s.position.x / PITCH_HALF_LENGTH)
    features.append(s.position.y / PITCH_HALF_WIDTH)
    features.append(s.velocity.x / MAX_PLAYER_SPEED)
    features.append(s.velocity.y / MAX_PLAYER_SPEED)
    features.append(s.stamina / MAX_STAMINA)
    features.append(s.body_direction / 180.0)
    features.append(s.relative_neck_direction / 180.0)
    features.append(1.0 if s.is_kickable else 0.0)

    # --- 2. Ball Information (6 dims) ---
    b = wm.ball
    features.append(b.position.x / PITCH_HALF_LENGTH)
    features.append(b.position.y / PITCH_HALF_WIDTH)
    features.append(b.velocity.x / MAX_BALL_SPEED)
    features.append(b.velocity.y / MAX_BALL_SPEED)
    features.append((b.position.x - s.position.x) / (PITCH_HALF_LENGTH * 2))
    features.append((b.position.y - s.position.y) / (PITCH_HALF_WIDTH * 2))

    # --- Helper to extract player features ---
    def _player_features(player_dict: Any, unum: int) -> list[float]:
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

    # --- 3. Teammates (1-11) (55 dims) ---
    for i in range(1, 12):
        features.extend(_player_features(wm.our_players_dict, i))

    # --- 4. Opponents (1-11) (55 dims) ---
    for i in range(1, 12):
        features.extend(_player_features(wm.their_players_dict, i))

    return np.array(features, dtype=np.float32)
