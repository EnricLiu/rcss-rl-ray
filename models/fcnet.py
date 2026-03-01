"""Custom fully-connected network for RLlib policy and value heads.

This model is registered under the name ``"rcss_fcnet"`` and can be
referenced by that string in any RLlib algorithm config.

Architecture (both policy and value share the same trunk):

    input → FC(256) → ReLU → FC(256) → ReLU
                                          ├─→ policy logits (action_space.n)
                                          └─→ value (1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict


class RCSSFCNet(TorchModelV2, nn.Module):
    """Shared-trunk fully-connected model for RCSS agents.

    The same trunk weights are used for both the policy (logits) head and
    the value-function head, keeping the model compact.

    Args:
        obs_space:     Observation space (expected to be a flat ``Box``).
        action_space:  Action space (expected to be a ``Discrete``).
        num_outputs:   Number of output logits (== action_space.n).
        model_config:  RLlib model config dict.  Recognised custom keys:

                       ``hidden_sizes`` (list[int], default ``[256, 256]``):
                           Width of each hidden layer in the trunk.
        name:          Model name used by RLlib.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Discrete,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hidden_sizes: list[int] = model_config.get(
            "custom_model_config", {}
        ).get("hidden_sizes", [256, 256])

        obs_dim = int(np.prod(obs_space.shape))

        # Build shared trunk.
        layers: list[nn.Module] = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        self._trunk = nn.Sequential(*layers)

        # Policy head (logits).
        self._policy_head = nn.Linear(in_size, num_outputs)
        # Value head.
        self._value_head = nn.Linear(in_size, 1)

        self._last_value: torch.Tensor | None = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """Compute action logits from *input_dict["obs_flat"]*."""
        obs = input_dict["obs_flat"].float()
        trunk_out = self._trunk(obs)
        self._last_value = self._value_head(trunk_out).squeeze(1)
        return self._policy_head(trunk_out), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value estimate computed during the last ``forward`` call."""
        assert self._last_value is not None, "forward() must be called first"
        return self._last_value


def register() -> None:
    """Register ``RCSSFCNet`` with RLlib's :class:`ModelCatalog`.

    Call this once before constructing any RLlib ``Algorithm`` that uses
    ``"custom_model": "rcss_fcnet"``.
    """
    ModelCatalog.register_custom_model("rcss_fcnet", RCSSFCNet)
