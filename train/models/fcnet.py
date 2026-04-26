from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.spaces import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from rcss_env.action_mask import ActionMaskResolver

class RCSSFCNet(TorchModelV2, nn.Module):

    _MASK_LOGIT_MIN = -1.0e9

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
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

        original_obs_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(original_obs_space, spaces.Dict) and "obs" in original_obs_space.spaces:
            feature_obs_space = original_obs_space["obs"]
        else:
            feature_obs_space = original_obs_space

        obs_dim = int(np.prod(feature_obs_space.shape))

        original_action_space = getattr(action_space, "original_space", action_space)
        self._discrete_action_dim = 0
        if isinstance(original_action_space, spaces.Dict) and "actions" in original_action_space.spaces:
            discrete_action_space = original_action_space["actions"]
            if isinstance(discrete_action_space, Discrete):
                self._discrete_action_dim = int(discrete_action_space.n)

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

        obs_payload = input_dict.get("obs")
        action_mask = None

        if isinstance(obs_payload, dict) and "obs" in obs_payload:
            obs = obs_payload["obs"].float()
            action_mask = obs_payload.get(ActionMaskResolver.OBSERVATION_KEY)
        else:
            obs = input_dict["obs_flat"].float()

        trunk_out = self._trunk(obs)
        self._last_value = self._value_head(trunk_out).squeeze(1)

        logits = self._policy_head(trunk_out)
        if action_mask is not None and self._discrete_action_dim > 0:
            mask = action_mask.float().clamp(0.0, 1.0)
            if mask.shape[-1] == self._discrete_action_dim and logits.shape[-1] >= self._discrete_action_dim:
                discrete_logits = logits[..., :self._discrete_action_dim]
                inf_mask = torch.where(
                    mask > 0,
                    torch.zeros_like(mask),
                    torch.full_like(mask, self._MASK_LOGIT_MIN),
                )
                logits = torch.cat(
                    [discrete_logits + inf_mask, logits[..., self._discrete_action_dim:]],
                    dim=-1,
                )

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:

        assert self._last_value is not None, "forward() must be called first"
        return self._last_value

def register() -> None:

    ModelCatalog.register_custom_model("rcss_fcnet", RCSSFCNet)
