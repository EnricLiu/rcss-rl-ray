from __future__ import annotations

import logging
from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


class RCSSCallbacks(DefaultCallbacks):
    REWARD_BREAKDOWN_TOTALS_KEY = "reward_breakdown_totals"
    REWARD_BREAKDOWN_STEPS_KEY = "reward_breakdown_steps"

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

        totals = episode.user_data.setdefault(self.REWARD_BREAKDOWN_TOTALS_KEY, {})
        for key, value in breakdown.items():
            totals[key] = float(totals.get(key, 0.0)) + float(value)
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
        episode.custom_metrics["episode_our_score"] = float(scores.get("our", 0))
        episode.custom_metrics["episode_their_score"] = float(scores.get("their", 0))
        step_value = last_info.get("step", episode.length)
        episode.custom_metrics["episode_steps"] = float(
            episode.length if step_value is None else step_value
        )

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
