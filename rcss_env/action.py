from __future__ import annotations

from typing import Any
from itertools import count
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .grpc_srv import pb2

@dataclass
class Action:
    action: int
    params: np.ndarray

    PARAM_LOW = -1.0
    PARAM_HIGH = 1.0

    ALL_ACTIONS = OrderedDict({
        "catch": {
            "cls": pb2.Catch,
        },
        "dash": {
            "cls": pb2.Dash,
            "power": (0.0, 100.0),
            "relative_direction": (-180.0, 180.0),
        },
        "kick": {
            "cls": pb2.Kick,
            "power": (0.0, 100.0),
            "relative_direction": (-180.0, 180.0),
        },
        "move": {
            "cls": pb2.Move,
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
        },
        "tackle": {
            "cls": pb2.Tackle,
            "power_or_dir": (0.0, 100.0),
            "foul": {0, 1},
        },
        "turn": {
            "cls": pb2.Turn,
            "relative_direction": (-180.0, 180.0),
        }
    })

    def __post_init__(self):
        if self.action < 0 or self.action >= len(self.ALL_ACTIONS):
            raise ValueError(f"Invalid action index: {self.action}. Must be in [0, {len(self.ALL_ACTIONS) - 1}].")

    def __get_action(self, cls, config: dict[str, Any]):
        pass

    def get_action(self):
        action_infos: list[dict[str, Any]] = list(self.ALL_ACTIONS.values())
        action_info = action_infos[self.action].copy()

        cls = action_info.pop("cls")
        param_start_idx = sum(len(info) - 1 for info in action_infos[:self.action])
        action_params = {}

        for param_idx, (param_name, constraints) in zip(count(start=param_start_idx), action_info.items()):
            param = self.params[param_idx]

            if isinstance(constraints, tuple) and len(constraints) == 2:
                low, high = constraints
                scaled_param = low + (param - self.PARAM_LOW) * (high - low) / (self.PARAM_HIGH - self.PARAM_LOW)
                action_params[param_name] = scaled_param

            elif isinstance(constraints, set):
                low, high = min(constraints), max(constraints)
                scaled_param = low + (param - self.PARAM_LOW) * (high - low) / (self.PARAM_HIGH - self.PARAM_LOW)

                closest_value = min(constraints, key=lambda x: abs(x - scaled_param))
                action_params[param_name] = closest_value

            else:
                raise ValueError(f"Invalid constraints for parameter '{param_name}': {constraints}")

        return cls(**action_params)

    @classmethod
    def n_actions(cls) -> int:
        return len(cls.ALL_ACTIONS.keys())

    @classmethod
    def n_action_params(cls) -> int:
        return sum(len(info) - 1 for info in cls.ALL_ACTIONS.values())

    @staticmethod
    def space_schema() -> spaces.Dict:
        return spaces.Dict({
            "actions": spaces.Discrete(Action.n_actions()),
            "params": spaces.Box(low=-1.0, high=1.0, shape=(Action.n_action_params(),), dtype=np.float32)
        })

    @staticmethod
    def from_space(action_dict: dict[str, Any]) -> Action:
        action = action_dict["actions"]
        params = action_dict["params"]
        return Action(action=action, params=params)
