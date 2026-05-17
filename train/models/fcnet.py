from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns

from rcss_env.action import Action
from rcss_env.action_mask import ActionMaskResolver


class RCSSPPOTorchRLModule(DefaultPPOTorchRLModule):
    """PPO RLModule that masks invalid RCSS discrete actions."""

    _MASK_LOGIT_MIN = -1.0e9

    def _split_masked_obs(
        self,
        batch: dict[str, Any],
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        obs_payload = batch.get(Columns.OBS)
        if (
            isinstance(obs_payload, Mapping)
            and "obs" in obs_payload
        ):
            model_batch = batch.copy()
            model_batch[Columns.OBS] = obs_payload["obs"]
            action_mask = obs_payload.get(ActionMaskResolver.OBSERVATION_KEY)
            return model_batch, action_mask

        return batch, None

    def _mask_action_dist_inputs(
        self,
        output: dict[str, Any],
        action_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        if action_mask is None or Columns.ACTION_DIST_INPUTS not in output:
            return output

        logits = output[Columns.ACTION_DIST_INPUTS]
        if logits.shape[-1] < Action.n_actions():
            return output

        mask = action_mask.to(
            device=logits.device,
            dtype=logits.dtype,
        ).clamp(0.0, 1.0)
        discrete_logits = logits[..., :Action.n_actions()]
        inf_mask = torch.where(
            mask > 0,
            torch.zeros_like(mask),
            torch.full_like(mask, self._MASK_LOGIT_MIN),
        )
        output[Columns.ACTION_DIST_INPUTS] = torch.cat(
            [
                discrete_logits + inf_mask,
                logits[..., Action.n_actions():],
            ],
            dim=-1,
        )
        return output

    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        model_batch, action_mask = self._split_masked_obs(batch)
        output = super()._forward(model_batch, **kwargs)
        return self._mask_action_dist_inputs(output, action_mask)

    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        model_batch, action_mask = self._split_masked_obs(batch)
        output = super()._forward_train(model_batch, **kwargs)
        return self._mask_action_dist_inputs(output, action_mask)

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Any | None = None,
    ) -> Any:
        model_batch, _ = self._split_masked_obs(batch)
        return super().compute_values(model_batch, embeddings=embeddings)
