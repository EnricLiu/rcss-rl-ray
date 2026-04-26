"""RCSS PPO RLModule with action masking (new API stack).

Migrated from the old ``TorchModelV2`` / ``ModelCatalog`` approach to the
RLlib new API stack (``RLModule`` / ``Learner`` / ``EnvRunner`` /
``ConnectorV2``).

The module expects a ``gymnasium.spaces.Dict`` observation space of the form::

    Dict({
        "obs":         Box(shape=(obs_dim,), dtype=float32),
        "action_mask": Box(low=0, high=1, shape=(n_discrete_actions,), dtype=int8),
    })

Action masking is applied to the *discrete* slice of ``ACTION_DIST_INPUTS``
so that logits for forbidden actions are set to ``-inf`` before sampling.
Continuous action parameters (the ``params`` part of the Dict action space)
are left unmasked.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import gymnasium as gym

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, _ = try_import_torch()

# Keys in the environment's Dict observation space
_OBS_KEY = "obs"
_MASK_KEY = "action_mask"


class RCSSPPORLModule(DefaultPPOTorchRLModule):
    """PPO RLModule for RCSS that applies action masking on the discrete head.

    The encoder receives only the ``"obs"`` sub-space (shape ``(obs_dim,)``),
    while the full Dict observation space (including ``"action_mask"``) is
    reported to callers via ``self.observation_space``.

    During each forward pass the ``"action_mask"`` tensor is extracted from
    the batch and used to clamp the discrete action logits to ``FLOAT_MIN``
    for disallowed actions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Intercept the observation_space so the encoder is built only on the
        # "obs" sub-space, not on the full Dict that includes the mask.
        obs_space: Optional[gym.Space] = kwargs.get("observation_space")
        if (
            obs_space is not None
            and isinstance(obs_space, gym.spaces.Dict)
            and _OBS_KEY in obs_space.spaces
        ):
            self._obs_space_with_mask = obs_space
            kwargs["observation_space"] = obs_space[_OBS_KEY]
        else:
            self._obs_space_with_mask = obs_space

        super().__init__(*args, **kwargs)

    @override(DefaultPPOTorchRLModule)
    def setup(self) -> None:
        super().setup()
        # After the encoder/pi/vf heads are built (which use the stripped obs
        # space), restore the full observation space so that this module
        # correctly reports the env's actual space.
        if self._obs_space_with_mask is not None:
            self.observation_space = self._obs_space_with_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_mask_and_strip_obs(
        self, batch: Dict[str, Any]
    ) -> tuple[Any, Dict[str, Any]]:
        """Pop the action mask from ``batch[OBS]`` and replace with raw obs.

        Returns ``(action_mask, modified_batch)`` where *modified_batch* is a
        shallow copy with ``Columns.OBS`` replaced by the ``"obs"`` tensor.
        """
        obs_dict = batch[Columns.OBS]
        action_mask = obs_dict[_MASK_KEY].float()
        batch = {**batch, Columns.OBS: obs_dict[_OBS_KEY]}
        return action_mask, batch

    def _apply_mask(
        self, out: Dict[str, Any], action_mask: Any
    ) -> Dict[str, Any]:
        """Add ``-inf`` to logits of disallowed discrete actions.

        Only the first ``n`` elements of ``ACTION_DIST_INPUTS`` are masked,
        where ``n = action_mask.shape[-1]`` (i.e. the number of discrete
        actions).  Continuous parameter logits that follow are left intact.
        """
        dist_inputs = out[Columns.ACTION_DIST_INPUTS]
        n = action_mask.shape[-1]
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_discrete = dist_inputs[..., :n] + inf_mask
        if dist_inputs.shape[-1] > n:
            out[Columns.ACTION_DIST_INPUTS] = torch.cat(
                [masked_discrete, dist_inputs[..., n:]], dim=-1
            )
        else:
            out[Columns.ACTION_DIST_INPUTS] = masked_discrete
        return out

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    @override(DefaultPPOTorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Inference / exploration forward pass with action masking."""
        action_mask, batch = self._extract_mask_and_strip_obs(batch)
        out = super()._forward(batch, **kwargs)
        return self._apply_mask(out, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """Training forward pass with action masking.

        The action mask is always expected to be present in ``batch[Columns.OBS]``
        when this is called from the PPO Learner with the raw training batch.
        The defensive ``isinstance`` check handles the edge case where the batch
        has already had its obs stripped (e.g. by ``compute_values``).
        """
        obs = batch.get(Columns.OBS)
        if isinstance(obs, dict) and _MASK_KEY in obs:
            action_mask, batch = self._extract_mask_and_strip_obs(batch)
        else:
            action_mask = None
        out = super()._forward_train(batch, **kwargs)
        if action_mask is not None:
            return self._apply_mask(out, action_mask)
        return out

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> Any:
        """Value function computation; strips action mask if still present."""
        obs = batch.get(Columns.OBS)
        if isinstance(obs, dict) and _MASK_KEY in obs:
            _, batch = self._extract_mask_and_strip_obs(batch)
        return super().compute_values(batch, embeddings)
