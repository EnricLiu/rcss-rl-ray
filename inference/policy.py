"""Strict per-unum MultiRLModule action computation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

from rcss_env import obs as observation
from rcss_env.action import Action
from rcss_env.action_mask import ActionMaskResolver

from .loader import BundleValidationError, LoadedBundle
from .manifest import BundleManifest


AgentObservation = Mapping[str, np.ndarray]
EnvironmentAction = dict[str, Any]


class PolicyInferenceError(RuntimeError):
    """Raised when inference inputs or outputs violate the runtime contract."""


def seed_inference(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_observation(
    agent_id: int,
    payload: AgentObservation,
) -> tuple[np.ndarray, np.ndarray]:
    if set(payload) != {"obs", ActionMaskResolver.OBSERVATION_KEY}:
        raise PolicyInferenceError(
            f"Agent {agent_id} observation keys must be exactly "
            f"{{'obs', '{ActionMaskResolver.OBSERVATION_KEY}'}}"
        )

    obs = np.asarray(payload["obs"])
    if obs.shape != (observation.dim(),) or obs.dtype != np.float32:
        raise PolicyInferenceError(
            f"Agent {agent_id} observation must be float32[{observation.dim()}], "
            f"got dtype={obs.dtype}, shape={obs.shape}"
        )
    if not np.isfinite(obs).all():
        raise PolicyInferenceError(f"Agent {agent_id} observation contains non-finite values")

    action_mask = np.asarray(payload[ActionMaskResolver.OBSERVATION_KEY])
    if action_mask.shape != (Action.n_actions(),) or action_mask.dtype != np.int8:
        raise PolicyInferenceError(
            f"Agent {agent_id} action_mask must be int8[{Action.n_actions()}], "
            f"got dtype={action_mask.dtype}, shape={action_mask.shape}"
        )
    if not np.isin(action_mask, (0, 1)).all():
        raise PolicyInferenceError(f"Agent {agent_id} action_mask must be binary")
    if not action_mask.any():
        raise PolicyInferenceError(f"Agent {agent_id} action_mask has no legal action")
    return obs, action_mask


def decode_action_sample(
    *,
    agent_id: int,
    sample: Mapping[str, torch.Tensor],
    action_mask: np.ndarray,
) -> EnvironmentAction:
    if set(sample) != {"actions", "params"}:
        raise PolicyInferenceError(
            f"Agent {agent_id} action sample has invalid keys: {sorted(sample)}"
        )

    discrete = sample["actions"].detach().cpu().numpy()
    params = sample["params"].detach().cpu().numpy()
    if discrete.shape != (1,):
        raise PolicyInferenceError(
            f"Agent {agent_id} discrete action must have shape (1,), got {discrete.shape}"
        )
    if params.shape != (1, Action.n_action_params()):
        raise PolicyInferenceError(
            f"Agent {agent_id} params must have shape "
            f"(1, {Action.n_action_params()}), got {params.shape}"
        )
    if not np.isfinite(params).all():
        raise PolicyInferenceError(f"Agent {agent_id} params contain non-finite values")

    action = int(discrete[0])
    if action < 0 or action >= Action.n_actions():
        raise PolicyInferenceError(f"Agent {agent_id} produced invalid action index {action}")
    if not Action.is_action_allowed(action, action_mask):
        raise PolicyInferenceError(
            f"Agent {agent_id} produced masked action {Action.action_name(action)!r}"
        )

    clipped = np.clip(
        params[0],
        Action.PARAM_LOW,
        Action.PARAM_HIGH,
    ).astype(np.float32, copy=False)
    return {"actions": action, "params": clipped}


@dataclass
class MultiAgentPolicyAdapter:
    module: MultiRLModule
    manifest: BundleManifest
    device: torch.device
    deterministic: bool = True

    @classmethod
    def from_loaded_bundle(
        cls,
        loaded: LoadedBundle,
        *,
        deterministic: bool | None = None,
    ) -> MultiAgentPolicyAdapter:
        if deterministic is None:
            deterministic = loaded.manifest.inference.default_mode == "deterministic"
        return cls(
            module=loaded.module,
            manifest=loaded.manifest,
            device=loaded.device,
            deterministic=deterministic,
        )

    def compute_actions(
        self,
        observations: Mapping[int, AgentObservation],
    ) -> dict[int, EnvironmentAction]:
        expected_agents = set(self.manifest.agent_ids)
        actual_agents = set(observations)
        if actual_agents != expected_agents:
            raise PolicyInferenceError(
                f"Observation agent set mismatch: expected={sorted(expected_agents)}, "
                f"actual={sorted(actual_agents)}"
            )

        module_batches: dict[str, dict[str, Any]] = {}
        agent_by_module: dict[str, int] = {}
        masks_by_agent: dict[int, np.ndarray] = {}
        mapping = self.manifest.policy_topology.agent_to_module

        for agent_id in sorted(observations):
            obs, action_mask = _validate_observation(
                agent_id,
                observations[agent_id],
            )
            module_id = mapping[str(agent_id)]
            module_batches[module_id] = {
                Columns.OBS: {
                    "obs": torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0),
                    ActionMaskResolver.OBSERVATION_KEY: torch.as_tensor(
                        action_mask,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0),
                }
            }
            agent_by_module[module_id] = agent_id
            masks_by_agent[agent_id] = action_mask

        try:
            with torch.inference_mode():
                outputs = self.module.forward_inference(module_batches)
        except Exception as exc:
            raise PolicyInferenceError("MultiRLModule forward_inference failed") from exc
        if set(outputs) != set(module_batches):
            raise PolicyInferenceError(
                f"Inference output module set mismatch: expected={sorted(module_batches)}, "
                f"actual={sorted(outputs)}"
            )

        actions: dict[int, EnvironmentAction] = {}
        for module_id, output in outputs.items():
            if Columns.ACTION_DIST_INPUTS not in output:
                raise PolicyInferenceError(
                    f"Module {module_id!r} did not return action_dist_inputs"
                )
            module = self.module[module_id]
            try:
                distribution = module.get_inference_action_dist_cls().from_logits(
                    output[Columns.ACTION_DIST_INPUTS]
                )
                if self.deterministic:
                    distribution = distribution.to_deterministic()
                sample = distribution.sample()
            except Exception as exc:
                raise PolicyInferenceError(
                    f"Module {module_id!r} action distribution failed"
                ) from exc
            agent_id = agent_by_module[module_id]
            actions[agent_id] = decode_action_sample(
                agent_id=agent_id,
                sample=sample,
                action_mask=masks_by_agent[agent_id],
            )

        if set(actions) != expected_agents:
            raise PolicyInferenceError("Decoded action agent set is incomplete")
        return actions


def adapter_from_bundle(
    loaded: LoadedBundle,
    *,
    deterministic: bool | None = None,
) -> MultiAgentPolicyAdapter:
    if set(loaded.module.keys()) != set(loaded.manifest.model.module_ids):
        raise BundleValidationError("Loaded bundle module topology is inconsistent")
    return MultiAgentPolicyAdapter.from_loaded_bundle(
        loaded,
        deterministic=deterministic,
    )
