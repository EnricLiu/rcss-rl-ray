from typing import Any

import numpy as np

from .grpc_srv import pb2

PITCH_HALF_LENGTH = 52.5
PITCH_HALF_WIDTH = 34.0
MAX_PLAYER_SPEED = 1.5
MAX_BALL_SPEED = 3.0
MAX_STAMINA = 8000.0

def dim() -> int:

    return 124

def extract(wm: pb2.WorldModel) -> np.ndarray:

    features: list[float] = []

    s = wm.self
    features.append(s.position.x / PITCH_HALF_LENGTH)
    features.append(s.position.y / PITCH_HALF_WIDTH)
    features.append(s.velocity.x / MAX_PLAYER_SPEED)
    features.append(s.velocity.y / MAX_PLAYER_SPEED)
    features.append(s.stamina / MAX_STAMINA)
    features.append(s.body_direction / 180.0)
    features.append(s.relative_neck_direction / 180.0)
    features.append(1.0 if s.is_kickable else 0.0)

    b = wm.ball
    features.append(b.position.x / PITCH_HALF_LENGTH)
    features.append(b.position.y / PITCH_HALF_WIDTH)
    features.append(b.velocity.x / MAX_BALL_SPEED)
    features.append(b.velocity.y / MAX_BALL_SPEED)
    features.append((b.position.x - s.position.x) / (PITCH_HALF_LENGTH * 2))
    features.append((b.position.y - s.position.y) / (PITCH_HALF_WIDTH * 2))

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

    for i in range(1, 12):
        features.extend(_player_features(wm.our_players_dict, i))

    for i in range(1, 12):
        features.extend(_player_features(wm.their_players_dict, i))

    return np.array(features, dtype=np.float32)
