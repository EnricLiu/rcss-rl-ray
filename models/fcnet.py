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

        layers: list[nn.Module] = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        self._trunk = nn.Sequential(*layers)

        self._policy_head = nn.Linear(in_size, num_outputs)

        self._value_head = nn.Linear(in_size, 1)

        self._last_value: torch.Tensor | None = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:

        obs = input_dict["obs_flat"].float()
        trunk_out = self._trunk(obs)
        self._last_value = self._value_head(trunk_out).squeeze(1)
        return self._policy_head(trunk_out), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:

        assert self._last_value is not None, "forward() must be called first"
        return self._last_value

def register() -> None:

    ModelCatalog.register_custom_model("rcss_fcnet", RCSSFCNet)
