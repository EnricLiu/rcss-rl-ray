"""RCSS hybrid discrete-continuous action definition.

Maps a discrete action index and a continuous parameter vector to a protobuf
PlayerAction message.  Supported action types: catch, dash, kick, move, tackle, turn.
"""

from __future__ import annotations

from typing import Any, Iterable
from itertools import count
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from cachetools import cached
from gymnasium import spaces

from .grpc_srv.proto import pb2


@dataclass
class Action:
    """A discrete action combined with continuous parameters.

    Attributes:
        action: Discrete action index in [0, n_actions()-1].
        params: Continuous parameter vector with values in [PARAM_LOW, PARAM_HIGH].
    """

    action: int
    params: np.ndarray

    # Parameter space bounds
    PARAM_LOW = -1.0
    PARAM_HIGH = 1.0

    # DO NOT CHANGE, should align with pb2 defination
    CATCH = "catch"
    DASH = "dash"
    KICK = "kick"
    MOVE = "move"
    TACKLE = "tackle"
    TURN = "turn"

    # All supported action types and their parameter constraints.
    # Each entry: key = action name, value = dict where "cls" is the protobuf
    # message class and the remaining key-value pairs map parameter names to
    # constraints (tuple = continuous range, set = discrete value set).
    ALL_ACTIONS = OrderedDict({
        CATCH: {
            "cls": pb2.Catch,
        },
        DASH: {
            "cls": pb2.Dash,
            "power": (0.0, 100.0),
            "relative_direction": (-180.0, 180.0),
        },
        KICK: {
            "cls": pb2.Kick,
            "power": (0.0, 100.0),
            "relative_direction": (-180.0, 180.0),
        },
        MOVE: {
            "cls": pb2.Move,
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
        },
        TACKLE: {
            "cls": pb2.Tackle,
            "power_or_dir": (0.0, 100.0),
            "foul": {0, 1},
        },
        TURN: {
            "cls": pb2.Turn,
            "relative_direction": (-180.0, 180.0),
        }
    })

    def __post_init__(self):
        if self.action < 0 or self.action >= len(self.ALL_ACTIONS):
            raise ValueError(f"Invalid action index: {self.action}. Must be in [0, {len(self.ALL_ACTIONS) - 1}].")

    def __get_action(self, cls, config: dict[str, Any]):
        pass

    @classmethod
    def action_names(cls) -> tuple[str, ...]:
        """Return the discrete action names in the exact encoded order."""
        return tuple(cls.ALL_ACTIONS.keys())

    @classmethod
    def action_index(cls, name: str) -> int:
        """Return the encoded discrete index for an action name."""
        try:
            return cls.action_names().index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown action name: {name}") from exc

    @classmethod
    def action_name(cls, index: int) -> str:
        """Return the action name for a discrete action index."""
        return cls.action_names()[index]

    def get_pb2_name(self) -> str:
        return self.action_names()[self.action]

    def get_action(self) -> pb2.Catch | pb2.Dash | pb2.Kick | pb2.Move | pb2.Tackle | pb2.Turn:
        """Convert the discrete index + continuous params into a protobuf action message.

        Each parameter component is linearly mapped from [PARAM_LOW, PARAM_HIGH] to
        the constraint range defined in ALL_ACTIONS.  For discrete constraint sets
        the closest legal value is selected.
        """
        action_infos: list[dict[str, Any]] = list(self.ALL_ACTIONS.values())
        action_info = action_infos[self.action].copy()

        cls = action_info.pop("cls")
        # Compute the offset of this action's params within the global param vector
        param_start_idx = sum(len(info) - 1 for info in action_infos[:self.action])
        action_params = {}

        for param_idx, (param_name, constraints) in zip(count(start=param_start_idx), action_info.items()):
            param = self.params[param_idx]

            if isinstance(constraints, tuple) and len(constraints) == 2:
                # Continuous range: linear map [PARAM_LOW, PARAM_HIGH] -> [low, high]
                low, high = constraints
                scaled_param = low + (param - self.PARAM_LOW) * (high - low) / (self.PARAM_HIGH - self.PARAM_LOW)
                action_params[param_name] = scaled_param

            elif isinstance(constraints, set):
                # Discrete value set: linear map then snap to closest legal value
                low, high = min(constraints), max(constraints)
                scaled_param = low + (param - self.PARAM_LOW) * (high - low) / (self.PARAM_HIGH - self.PARAM_LOW)

                closest_value = min(constraints, key=lambda x: abs(x - scaled_param))
                action_params[param_name] = closest_value

            else:
                raise ValueError(f"Invalid constraints for parameter '{param_name}': {constraints}")

        return cls(**action_params)

    def to_player_action(self) -> pb2.PlayerAction:
        return pb2.PlayerAction(**{ self.get_pb2_name(): self.get_action() })

    @classmethod
    @cached(cache={})
    def n_actions(cls) -> int:
        """Return the total number of discrete actions."""
        return len(cls.action_names())

    @classmethod
    def n_action_params(cls) -> int:
        """Return the total parameter dimension (sum of param counts across all action types)."""
        return sum(len(info) - 1 for info in cls.ALL_ACTIONS.values())

    @staticmethod
    def space_schema() -> spaces.Dict:
        """Build the Gymnasium action space: Discrete(n_actions) + Box(n_action_params)."""
        return spaces.Dict({
            "actions": spaces.Discrete(Action.n_actions()),
            "params": spaces.Box(low=-1.0, high=1.0, shape=(Action.n_action_params(),), dtype=np.float32)
        })

    @staticmethod
    def from_space(action_dict: dict[str, Any]) -> Action:
        """Construct an Action from a Gymnasium space sample dictionary."""
        action = action_dict["actions"]
        params = action_dict["params"]
        return Action(action=action, params=params)

    @classmethod
    def mask_from_allowed(cls, names: Iterable[str]) -> np.ndarray:
        """Build an ``act_mask`` vector from allowed action names."""
        allowed = set(names)
        return np.asarray(
            [1 if action_name in allowed else 0 for action_name in cls.action_names()],
            dtype=np.int8,
        )

    @classmethod
    def mask_from_blocked(cls, names: Iterable[str]) -> np.ndarray:
        """Build an ``act_mask`` vector from blocked action names."""
        blocked = set(names)
        return np.asarray(
            [0 if action_name in blocked else 1 for action_name in cls.action_names()],
            dtype=np.int8,
        )

    @classmethod
    def full_action_mask(cls) -> np.ndarray:
        """Return a permissive mask where all discrete actions are enabled."""
        return np.ones(cls.n_actions(), dtype=np.int8)

    @classmethod
    def is_action_allowed(cls, action: int, act_mask: np.ndarray | None) -> bool:
        """Check whether a discrete action index is enabled by the given mask."""
        if act_mask is None:
            return True
        if action < 0 or action >= len(act_mask):
            return False
        return bool(np.asarray(act_mask)[action])

