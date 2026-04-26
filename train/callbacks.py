from __future__ import annotations

import logging
from numbers import Number
from typing import Any, cast

from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


class RCSSCallbacks(DefaultCallbacks):
    REWARD_BREAKDOWN_TOTALS_KEY = "reward_breakdown_totals"
    REWARD_BREAKDOWN_STEPS_KEY = "reward_breakdown_steps"
    CHECKPOINT_SCORE_ATTRIBUTE = "checkpoint_score"
    CHECKPOINT_SCORE_SOURCE_ATTRIBUTE = "env_runners/episode_reward_mean"

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

    def on_train_result(
        self,
        *,
        algorithm: Any,
        result: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        score = self._lookup_metric(result, self.CHECKPOINT_SCORE_SOURCE_ATTRIBUTE)
        score_value = self._coerce_float(score)
        if score_value is not None:
            result[self.CHECKPOINT_SCORE_ATTRIBUTE] = score_value

    def on_episode_start(
        self,
        *,
        episode: Any,
        **kwargs: Any,
    ) -> None:
        episode.user_data[self.REWARD_BREAKDOWN_TOTALS_KEY] = {}
        episode.user_data[self.REWARD_BREAKDOWN_STEPS_KEY] = 0

    def on_episode_step(
        self,
        *,
        episode: Any,
        **kwargs: Any,
    ) -> None:
        breakdown: dict[str, Any] | None = None
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info and "reward_breakdown" in info:
                breakdown = info["reward_breakdown"]
                break

        if not breakdown:
            return

        totals: dict[str, float] = episode.user_data.setdefault(
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
        episode.user_data[self.REWARD_BREAKDOWN_STEPS_KEY] = int(
            episode.user_data.get(self.REWARD_BREAKDOWN_STEPS_KEY, 0)
        ) + 1

    def on_episode_end(
        self,
        *,
        episode: Any,
        **kwargs: Any,
    ) -> None:

        last_info: dict[str, Any] = {}
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info and "scores" in info:
                last_info = info
                break

        scores = last_info.get("scores", {"our": 0, "their": 0})
        our_score = self._coerce_float(scores.get("our", 0))
        their_score = self._coerce_float(scores.get("their", 0))
        episode.custom_metrics["episode_our_score"] = 0.0 if our_score is None else our_score
        episode.custom_metrics["episode_their_score"] = 0.0 if their_score is None else their_score
        step_value = last_info.get("step", episode.length)
        episode.custom_metrics["episode_steps"] = self._coerce_float(
            episode.length if step_value is None else step_value
        ) or 0.0

        reward_breakdown_totals = episode.user_data.get(self.REWARD_BREAKDOWN_TOTALS_KEY, {})
        reward_breakdown_steps = int(episode.user_data.get(self.REWARD_BREAKDOWN_STEPS_KEY, 0))
        for key, total in reward_breakdown_totals.items():
            metric_base = f"reward_{key}"
            total_value = float(total)
            episode.custom_metrics[f"{metric_base}_total"] = total_value
            if reward_breakdown_steps > 0:
                episode.custom_metrics[f"{metric_base}_per_step"] = total_value / reward_breakdown_steps

        logger.debug(
            "Episode finished | steps=%d | our=%d | their=%d | reward_metrics=%s",
            episode.length,
            scores.get("our", 0),
            scores.get("their", 0),
            sorted(reward_breakdown_totals.keys()),
        )
