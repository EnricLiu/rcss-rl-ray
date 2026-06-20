"""Episode orchestration for loaded inference policies and RCSSEnv."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from rcss_env.env import RCSSEnv
from rcss_env.action import Action

from .config import InferenceConfig
from .loader import LoadedBundle, load_bundle, validate_curriculum_agents
from .policy import MultiAgentPolicyAdapter, seed_inference

logger = logging.getLogger(__name__)


def _latency_summary(values: tuple[float, ...] | list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        return ordered[index]

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


class PolicyLike(Protocol):
    def compute_actions(self, observations: Mapping[int, Any]) -> dict[int, Any]: ...


class EnvironmentLike(Protocol):
    def reset(self, *, seed: int | None = None, options: Any = None) -> Any: ...
    def step(self, actions: dict[int, Any]) -> Any: ...
    def close(self) -> None: ...


class RunnerInfrastructureError(RuntimeError):
    """Raised when reset/step communication retries are exhausted."""


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    attempts: int
    steps: int
    termination_reason: str
    rewards: dict[int, float]
    our_score: float | None = None
    their_score: float | None = None
    final_cycle: int | None = None
    episode_seconds: float = 0.0
    decision_latency_seconds: tuple[float, ...] = ()
    action_counts: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def decision_latency_summary(self) -> dict[str, float | int]:
        return _latency_summary(self.decision_latency_seconds)


@dataclass(frozen=True)
class RunSummary:
    episodes: tuple[EpisodeResult, ...]
    interrupted: bool = False
    requested_episodes: int = 0

    @property
    def metrics(self) -> dict[str, Any]:
        decision_latencies = tuple(
            latency
            for episode in self.episodes
            for latency in episode.decision_latency_seconds
        )
        actions_total: dict[str, int] = {}
        for episode in self.episodes:
            for counts in episode.action_counts.values():
                for action_name, count in counts.items():
                    actions_total[action_name] = actions_total.get(action_name, 0) + count
        completed = sum(
            episode.termination_reason != "interrupted" for episode in self.episodes
        )
        return {
            "episodes_requested": self.requested_episodes,
            "episodes_completed": completed,
            "infrastructure_retries": sum(
                max(0, episode.attempts - 1) for episode in self.episodes
            ),
            "episode_seconds_total": sum(
                episode.episode_seconds for episode in self.episodes
            ),
            "episode_steps_total": sum(episode.steps for episode in self.episodes),
            "decision_latency_seconds": _latency_summary(decision_latencies),
            "actions_total": actions_total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "interrupted": self.interrupted,
            "metrics": self.metrics,
            "episodes": [
                {
                    "episode_index": episode.episode_index,
                    "attempts": episode.attempts,
                    "steps": episode.steps,
                    "termination_reason": episode.termination_reason,
                    "rewards": episode.rewards,
                    "our_score": episode.our_score,
                    "their_score": episode.their_score,
                    "final_cycle": episode.final_cycle,
                    "episode_seconds": episode.episode_seconds,
                    "decision_latency_seconds": episode.decision_latency_summary,
                    "action_counts": episode.action_counts,
                }
                for episode in self.episodes
            ],
        }


def _has_reset_needed(infos: Mapping[int, Mapping[str, Any]]) -> bool:
    return any(bool(info.get("reset_needed")) for info in infos.values())


def _latest_scores(
    infos: Mapping[int, Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    for info in infos.values():
        scores = info.get("scores")
        if isinstance(scores, Mapping):
            our = scores.get("our")
            their = scores.get("their")
            return (
                None if our is None else float(our),
                None if their is None else float(their),
            )
    return None, None


def _latest_cycle(infos: Mapping[int, Mapping[str, Any]]) -> int | None:
    for info in infos.values():
        cycle = info.get("cycle")
        if cycle is not None:
            return int(cycle)
    return None


@dataclass
class InferenceRunner:
    env: EnvironmentLike
    policy: PolicyLike
    episodes: int
    seed: int = 0
    max_episode_retries: int = 0
    trace_actions: bool = False
    stop_requested: Callable[[], bool] = field(default=lambda: False)

    def run(self) -> RunSummary:
        results: list[EpisodeResult] = []
        interrupted = False
        try:
            for episode_index in range(self.episodes):
                if self.stop_requested():
                    interrupted = True
                    break
                result = self._run_episode_with_retries(episode_index)
                if result is None:
                    interrupted = True
                    break
                results.append(result)
                if result.termination_reason == "interrupted":
                    interrupted = True
                    break
            return RunSummary(
                episodes=tuple(results),
                interrupted=interrupted,
                requested_episodes=self.episodes,
            )
        finally:
            self.env.close()

    def _run_episode_with_retries(
        self,
        episode_index: int,
    ) -> EpisodeResult | None:
        last_error: BaseException | None = None
        for attempt_index in range(self.max_episode_retries + 1):
            if self.stop_requested():
                return None
            try:
                observation, _ = self.env.reset(seed=self.seed + episode_index)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Episode %d reset attempt %d/%d failed: %s",
                    episode_index,
                    attempt_index + 1,
                    self.max_episode_retries + 1,
                    last_error,
                )
                continue

            try:
                result = self._run_started_episode(
                    episode_index=episode_index,
                    attempts=attempt_index + 1,
                    observation=observation,
                )
                if result is not None:
                    return result
                last_error = RunnerInfrastructureError(
                    "Environment returned reset_needed during step"
                )
            except RunnerInfrastructureError as exc:
                last_error = exc

            logger.warning(
                "Episode %d infrastructure attempt %d/%d failed: %s",
                episode_index,
                attempt_index + 1,
                self.max_episode_retries + 1,
                last_error,
            )

        raise RunnerInfrastructureError(
            f"Episode {episode_index} failed after "
            f"{self.max_episode_retries + 1} attempt(s)"
        ) from last_error

    def _run_started_episode(
        self,
        *,
        episode_index: int,
        attempts: int,
        observation: Mapping[int, Any],
    ) -> EpisodeResult | None:
        steps = 0
        started_at = time.perf_counter()
        decision_latencies: list[float] = []
        action_counts = {agent_id: {} for agent_id in observation}
        reward_totals = {agent_id: 0.0 for agent_id in observation}
        while True:
            if self.stop_requested():
                return EpisodeResult(
                    episode_index=episode_index,
                    attempts=attempts,
                    steps=steps,
                    termination_reason="interrupted",
                    rewards=reward_totals,
                    episode_seconds=time.perf_counter() - started_at,
                    decision_latency_seconds=tuple(decision_latencies),
                    action_counts=action_counts,
                )

            decision_started_at = time.perf_counter()
            actions = self.policy.compute_actions(observation)
            decision_latencies.append(time.perf_counter() - decision_started_at)
            for agent_id, action in actions.items():
                action_name = Action.action_name(int(action["actions"]))
                counts = action_counts.setdefault(agent_id, {})
                counts[action_name] = counts.get(action_name, 0) + 1
            if self.trace_actions:
                logger.info(
                    "Episode %d step %d actions=%s decision_latency_seconds=%.6f",
                    episode_index,
                    steps,
                    actions,
                    decision_latencies[-1],
                )
            try:
                observation, rewards, terminated, truncated, infos = self.env.step(actions)
            except Exception as exc:
                raise RunnerInfrastructureError("Environment step failed") from exc
            steps += 1
            for agent_id, reward in rewards.items():
                reward_totals[agent_id] = reward_totals.get(agent_id, 0.0) + float(reward)

            if _has_reset_needed(infos):
                return None

            terminated_all = bool(terminated.get("__all__"))
            truncated_all = bool(truncated.get("__all__"))
            if terminated_all or truncated_all:
                our_score, their_score = _latest_scores(infos)
                final_cycle = _latest_cycle(infos)
                reason = "terminated" if terminated_all else "time_limit"
                return EpisodeResult(
                    episode_index=episode_index,
                    attempts=attempts,
                    steps=steps,
                    termination_reason=reason,
                    rewards=reward_totals,
                    our_score=our_score,
                    their_score=their_score,
                    final_cycle=final_cycle,
                    episode_seconds=time.perf_counter() - started_at,
                    decision_latency_seconds=tuple(decision_latencies),
                    action_counts=action_counts,
                )


def execute_inference(
    config: InferenceConfig,
    *,
    env_factory: Callable[[Any], EnvironmentLike] = RCSSEnv,
    bundle_loader: Callable[..., LoadedBundle] = load_bundle,
    stop_requested: Callable[[], bool] = lambda: False,
) -> RunSummary:
    seed_inference(config.seed)
    loaded = bundle_loader(
        config.bundle_path,
        device=config.device,
    )
    env_config = config.build_env_config()
    validate_curriculum_agents(
        loaded.manifest,
        env_config.curriculum.agent_unums(),
    )
    policy = MultiAgentPolicyAdapter.from_loaded_bundle(
        loaded,
        deterministic=config.deterministic,
    )
    logger.info(
        "Loaded inference model=%s version=%s source=%s git_commit=%s "
        "device=%s deterministic=%s curriculum=%s mapping=%s",
        loaded.manifest.model_name,
        loaded.manifest.model_version,
        loaded.manifest.source.checkpoint_uri,
        loaded.manifest.source.git_commit,
        loaded.device,
        policy.deterministic,
        config.curriculum_config.type,
        loaded.manifest.policy_topology.agent_to_module,
    )
    env = env_factory(env_config)
    return InferenceRunner(
        env=env,
        policy=policy,
        episodes=config.episodes,
        seed=config.seed,
        max_episode_retries=config.max_episode_retries,
        trace_actions=config.logging.trace_actions,
        stop_requested=stop_requested,
    ).run()
