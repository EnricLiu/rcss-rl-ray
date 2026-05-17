from __future__ import annotations

import logging
from numbers import Number
from typing import Any, cast

from ray.rllib.callbacks.callbacks import RLlibCallback

logger = logging.getLogger(__name__)


class RCSSCallbacks(RLlibCallback):
    REWARD_BREAKDOWN_TOTALS_KEY = "reward_breakdown_totals"
    REWARD_BREAKDOWN_STEPS_KEY = "reward_breakdown_steps"
    CHECKPOINT_SCORE_ATTRIBUTE = "checkpoint_score"
    CHECKPOINT_SCORE_SOURCE_ATTRIBUTE = "env_runners/episode_return_mean"
    CHECKPOINT_SCORE_FALLBACK_ATTRIBUTES = (
        "env_runners/episode_return_mean",
        "env_runners/episode_reward_mean",
        "episode_return_mean",
        "episode_reward_mean",
    )

    @staticmethod
    def _lookup_metric(result: dict[str, Any], metric: str) -> Any:
        if metric in result:
            return result[metric]

        current: Any = result
        for part in metric.split("/"):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _set_metric_if_missing(result: dict[str, Any], metric: str, value: float) -> None:
        if metric in result:
            return
        result[metric] = value

    @staticmethod
    def _episode_data(episode: Any) -> dict[str, Any]:
        if hasattr(episode, "user_data"):
            return cast(dict[str, Any], episode.user_data)
        return cast(dict[str, Any], episode.custom_data)

    @staticmethod
    def _latest_infos(episode: Any) -> list[dict[str, Any]]:
        if hasattr(episode, "get_agents") and hasattr(episode, "last_info_for"):
            return [
                info
                for agent_id in episode.get_agents()
                if (info := episode.last_info_for(agent_id))
            ]

        infos = episode.get_infos(indices=-1)
        if isinstance(infos, dict):
            return [
                info
                for info in infos.values()
                if isinstance(info, dict)
            ]
        return []

    @classmethod
    def _record_metric(
        cls,
        episode: Any,
        metrics_logger: Any | None,
        key: str,
        value: float,
    ) -> None:
        if hasattr(episode, "custom_metrics"):
            episode.custom_metrics[key] = value
        if metrics_logger is not None:
            metrics_logger.log_value(("custom_metrics", key), value)

    @staticmethod
    def _episode_length(episode: Any) -> int | float:
        if hasattr(episode, "length"):
            return episode.length
        return getattr(episode, "env_steps", 0)

    def on_train_result(
        self,
        *,
        algorithm: Any,
        result: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        score_value: float | None = None
        for metric in (
            self.CHECKPOINT_SCORE_SOURCE_ATTRIBUTE,
            *self.CHECKPOINT_SCORE_FALLBACK_ATTRIBUTES,
        ):
            score = self._lookup_metric(result, metric)
            score_value = self._coerce_float(score)
            if score_value is not None:
                break

        final_score = score_value
        if final_score is None:
            # RLlib's new API stack may produce one or more initial training
            # results before any episode has completed. Tune validates the
            # configured metric on every result, so always expose finite scores.
            final_score = 0.0

        result[self.CHECKPOINT_SCORE_ATTRIBUTE] = final_score
        self._set_metric_if_missing(
            result,
            self.CHECKPOINT_SCORE_SOURCE_ATTRIBUTE,
            final_score,
        )

    def on_episode_start(
        self,
        *,
        episode: Any,
        **kwargs: Any,
    ) -> None:
        data = self._episode_data(episode)
        data[self.REWARD_BREAKDOWN_TOTALS_KEY] = {}
        data[self.REWARD_BREAKDOWN_STEPS_KEY] = 0

    def on_episode_step(
        self,
        *,
        episode: Any,
        **kwargs: Any,
    ) -> None:
        breakdown: dict[str, Any] | None = None
        for info in self._latest_infos(episode):
            if info and "reward_breakdown" in info:
                breakdown = info["reward_breakdown"]
                break

        if not breakdown:
            return

        data = self._episode_data(episode)
        totals: dict[str, float] = data.setdefault(
            self.REWARD_BREAKDOWN_TOTALS_KEY, {}
        )
        for key, value in breakdown.items():
            if not isinstance(value, Number):
                continue
            current_total = totals[key] if key in totals else 0.0
            numeric_value = self._coerce_float(cast(Number, value))
            if numeric_value is None:
                continue
            totals[key] = current_total + numeric_value
        data[self.REWARD_BREAKDOWN_STEPS_KEY] = int(
            data.get(self.REWARD_BREAKDOWN_STEPS_KEY, 0)
        ) + 1

    def on_episode_end(
        self,
        *,
        episode: Any,
        metrics_logger: Any | None = None,
        **kwargs: Any,
    ) -> None:

        last_info: dict[str, Any] = {}
        for info in self._latest_infos(episode):
            if info and "scores" in info:
                last_info = info
                break

        scores = last_info.get("scores", {"our": 0, "their": 0})
        our_score = self._coerce_float(scores.get("our", 0))
        their_score = self._coerce_float(scores.get("their", 0))
        self._record_metric(
            episode,
            metrics_logger,
            "episode_our_score",
            0.0 if our_score is None else our_score,
        )
        self._record_metric(
            episode,
            metrics_logger,
            "episode_their_score",
            0.0 if their_score is None else their_score,
        )
        step_value = last_info.get("step", self._episode_length(episode))
        self._record_metric(
            episode,
            metrics_logger,
            "episode_steps",
            self._coerce_float(
                self._episode_length(episode) if step_value is None else step_value
            ) or 0.0,
        )

        data = self._episode_data(episode)
        reward_breakdown_totals = data.get(self.REWARD_BREAKDOWN_TOTALS_KEY, {})
        reward_breakdown_steps = int(data.get(self.REWARD_BREAKDOWN_STEPS_KEY, 0))
        for key, total in reward_breakdown_totals.items():
            metric_base = f"reward_{key}"
            total_value = float(total)
            self._record_metric(
                episode,
                metrics_logger,
                f"{metric_base}_total",
                total_value,
            )
            if reward_breakdown_steps > 0:
                self._record_metric(
                    episode,
                    metrics_logger,
                    f"{metric_base}_per_step",
                    total_value / reward_breakdown_steps,
                )

        logger.debug(
            "Episode finished | steps=%d | our=%d | their=%d | reward_metrics=%s",
            self._episode_length(episode),
            scores.get("our", 0),
            scores.get("their", 0),
            sorted(reward_breakdown_totals.keys()),
        )

from typing import Optional, Union, List
from aim.sdk import Repo, Run
from ray.tune.experiment import Trial
from ray.tune.logger.aim import AimLoggerCallback

class AimCallback(AimLoggerCallback):
    def __init__(
        self,
        repo: Optional[Union[str, "Repo"]] = None,
        experiment_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        run_params: Optional[dict[str, Any]] = None,
        **aim_run_kwargs,
    ):
        """
        See help(AimLoggerCallback) for more information about parameters.
        """
        super().__init__(
            repo=repo,
            experiment_name=experiment_name,
            metrics=metrics,
            **aim_run_kwargs,
        )
        self._run_params = run_params or {}

    def _create_run(self, trial: "Trial") -> Run:
        run = super()._create_run(trial)
        for key, value in self._run_params.items():
            setattr(run, key, value)
        return run
