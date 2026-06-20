"""Per-unum MultiRLModule inference with optional match-safe fallback actions."""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)


AgentObservation = Mapping[str, np.ndarray]
EnvironmentAction = dict[str, Any]


class PolicyInferenceError(RuntimeError):
    """Raised when inference inputs or outputs violate the runtime contract."""


def fallback_action(action_mask: np.ndarray | None = None) -> EnvironmentAction:
    """Return a low-impact action that is legal under the best available mask."""
    # TURN is never blocked by the current ActionMaskResolver. A zero-degree turn
    # keeps the command stream alive without moving the player or touching the ball.
    preferred = (Action.TURN, Action.DASH, Action.KICK, Action.TACKLE, Action.CATCH)
    selected = Action.TURN
    if action_mask is not None:
        for name in preferred:
            index = Action.action_index(name)
            if Action.is_action_allowed(index, action_mask):
                selected = name
                break
        else:
            raise PolicyInferenceError("Cannot build fallback action: no legal action")

    if selected == Action.TURN:
        action = Action.turn(relative_direction=0.0)
    elif selected == Action.DASH:
        action = Action.dash(power=0.0, relative_direction=0.0)
    elif selected == Action.KICK:
        action = Action.kick(power=0.0, relative_direction=0.0)
    elif selected == Action.TACKLE:
        action = Action.tackle(power_or_dir=0.0, foul=False)
    else:
        action = Action.catch()
    return {
        "actions": action.action,
        "params": action.params.astype(np.float32, copy=False),
    }


def _usable_fallback_mask(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, Mapping):
        return None
    raw_mask = payload.get(ActionMaskResolver.OBSERVATION_KEY)
    if raw_mask is None:
        return None
    mask = np.asarray(raw_mask)
    if (
        mask.shape != (Action.n_actions(),)
        or not np.isin(mask, (0, 1)).all()
        or not mask.any()
    ):
        return None
    return mask


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
    if not np.array_equal(clipped, params[0]):
        logger.debug("Agent %d action parameters were clipped to [-1, 1]", agent_id)
    return {"actions": action, "params": clipped}


@dataclass
class MultiAgentPolicyAdapter:
    module: MultiRLModule
    manifest: BundleManifest
    device: torch.device
    deterministic: bool = True
    fallback_on_error: bool = False
    _fallback_streaks: dict[int, int] = field(default_factory=dict, init=False)
    _forward_failure_streak: int = field(default=0, init=False)

    @classmethod
    def from_loaded_bundle(
        cls,
        loaded: LoadedBundle,
        *,
        deterministic: bool | None = None,
        fallback_on_error: bool = False,
    ) -> MultiAgentPolicyAdapter:
        if deterministic is None:
            deterministic = loaded.manifest.inference.default_mode == "deterministic"
        return cls(
            module=loaded.module,
            manifest=loaded.manifest,
            device=loaded.device,
            deterministic=deterministic,
            fallback_on_error=fallback_on_error,
        )

    def _fallback(
        self,
        agent_id: int,
        payload: Any,
        reason: str,
        *,
        level: int = logging.WARNING,
    ) -> EnvironmentAction:
        streak = self._fallback_streaks.get(agent_id, 0) + 1
        self._fallback_streaks[agent_id] = streak
        log_level = level if streak == 1 or streak % 100 == 0 else logging.DEBUG
        logger.log(
            log_level,
            "Agent %d using fallback action (consecutive=%d): %s",
            agent_id,
            streak,
            reason,
        )
        return fallback_action(_usable_fallback_mask(payload))

    def _mark_agent_success(self, agent_id: int) -> None:
        streak = self._fallback_streaks.pop(agent_id, 0)
        if streak:
            logger.info(
                "Agent %d policy inference recovered after %d fallback action(s)",
                agent_id,
                streak,
            )

    def _fallback_after_forward_failure(
        self,
        agent_by_module: Mapping[str, int],
        masks_by_agent: Mapping[int, np.ndarray],
        actions: dict[int, EnvironmentAction],
        reason: str,
    ) -> dict[int, EnvironmentAction]:
        self._forward_failure_streak += 1
        level = (
            logging.ERROR
            if self._forward_failure_streak == 1
            or self._forward_failure_streak % 100 == 0
            else logging.DEBUG
        )
        logger.log(
            level,
            "MultiRLModule forward failed (consecutive=%d); all affected agents "
            "will use fallback actions: %s",
            self._forward_failure_streak,
            reason,
        )
        for agent_id in agent_by_module.values():
            actions[agent_id] = fallback_action(masks_by_agent[agent_id])
        return actions

    def compute_actions(
        self,
        observations: Mapping[int, AgentObservation],
    ) -> dict[int, EnvironmentAction]:
        expected_agents = set(self.manifest.agent_ids)
        actual_agents = set(observations)
        if actual_agents != expected_agents:
            message = (
                f"Observation agent set mismatch: expected={sorted(expected_agents)}, "
                f"actual={sorted(actual_agents)}"
            )
            if not self.fallback_on_error:
                raise PolicyInferenceError(message)
            logger.warning("%s; missing agents will use fallback", message)

        module_batches: dict[str, dict[str, Any]] = {}
        agent_by_module: dict[str, int] = {}
        masks_by_agent: dict[int, np.ndarray] = {}
        actions: dict[int, EnvironmentAction] = {}
        mapping = self.manifest.policy_topology.agent_to_module

        for agent_id in sorted(expected_agents):
            payload = observations.get(agent_id)
            if payload is None:
                actions[agent_id] = self._fallback(
                    agent_id, None, "observation is missing"
                )
                continue
            try:
                obs, action_mask = _validate_observation(agent_id, payload)
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
            except Exception as exc:
                if not self.fallback_on_error:
                    if isinstance(exc, PolicyInferenceError):
                        raise
                    raise PolicyInferenceError(
                        f"Agent {agent_id} observation tensor conversion failed"
                    ) from exc
                actions[agent_id] = self._fallback(agent_id, payload, str(exc))
                continue
            agent_by_module[module_id] = agent_id
            masks_by_agent[agent_id] = action_mask

        outputs: Mapping[str, Mapping[str, torch.Tensor]] = {}
        if module_batches:
            try:
                with torch.inference_mode():
                    outputs = self.module.forward_inference(module_batches)
            except Exception as exc:
                if not self.fallback_on_error:
                    raise PolicyInferenceError(
                        "MultiRLModule forward_inference failed"
                    ) from exc
                return self._fallback_after_forward_failure(
                    agent_by_module,
                    masks_by_agent,
                    actions,
                    str(exc),
                )

        if self._forward_failure_streak:
            logger.info(
                "MultiRLModule forward recovered after %d failed decision(s)",
                self._forward_failure_streak,
            )
            self._forward_failure_streak = 0

        if not isinstance(outputs, Mapping):
            if not self.fallback_on_error:
                raise PolicyInferenceError("MultiRLModule output must be a mapping")
            return self._fallback_after_forward_failure(
                agent_by_module,
                masks_by_agent,
                actions,
                f"returned {type(outputs).__name__} instead of a mapping",
            )

        if set(outputs) != set(module_batches):
            message = (
                f"Inference output module set mismatch: expected={sorted(module_batches)}, "
                f"actual={sorted(outputs)}"
            )
            if not self.fallback_on_error:
                raise PolicyInferenceError(message)
            unexpected = set(outputs) - set(module_batches)
            if unexpected:
                logger.warning("Ignoring unexpected module outputs: %s", sorted(unexpected))

        for module_id in module_batches:
            agent_id = agent_by_module[module_id]
            output = outputs.get(module_id)
            if output is None:
                actions[agent_id] = self._fallback(
                    agent_id,
                    observations.get(agent_id),
                    f"module {module_id!r} returned no output",
                )
                continue
            if Columns.ACTION_DIST_INPUTS not in output:
                message = f"Module {module_id!r} did not return action_dist_inputs"
                if not self.fallback_on_error:
                    raise PolicyInferenceError(message)
                actions[agent_id] = self._fallback(
                    agent_id, observations.get(agent_id), message
                )
                continue
            module = self.module[module_id]
            try:
                distribution = module.get_inference_action_dist_cls().from_logits(
                    output[Columns.ACTION_DIST_INPUTS]
                )
                if self.deterministic:
                    distribution = distribution.to_deterministic()
                sample = distribution.sample()
                actions[agent_id] = decode_action_sample(
                    agent_id=agent_id,
                    sample=sample,
                    action_mask=masks_by_agent[agent_id],
                )
                self._mark_agent_success(agent_id)
            except Exception as exc:
                if not self.fallback_on_error:
                    if isinstance(exc, PolicyInferenceError):
                        raise
                    raise PolicyInferenceError(
                        f"Module {module_id!r} action distribution failed"
                    ) from exc
                actions[agent_id] = self._fallback(
                    agent_id,
                    observations.get(agent_id),
                    f"module {module_id!r} output was invalid: {exc}",
                )

        if set(actions) != expected_agents:
            raise PolicyInferenceError("Decoded action agent set is incomplete")
        return actions


def adapter_from_bundle(
    loaded: LoadedBundle,
    *,
    deterministic: bool | None = None,
    fallback_on_error: bool = False,
) -> MultiAgentPolicyAdapter:
    if set(loaded.module.keys()) != set(loaded.manifest.model.module_ids):
        raise BundleValidationError("Loaded bundle module topology is inconsistent")
    return MultiAgentPolicyAdapter.from_loaded_bundle(
        loaded,
        deterministic=deterministic,
        fallback_on_error=fallback_on_error,
    )
