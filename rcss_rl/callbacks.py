"""RLlib training callbacks for logging and checkpointing.

Callbacks are invoked by RLlib at various points in the training loop.
Subclass :class:`RCSSCallbacks` to add custom logic (e.g. environment
metrics, external logging, curriculum adjustments).
"""

from __future__ import annotations

import logging
from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

logger = logging.getLogger(__name__)


class RCSSCallbacks(DefaultCallbacks):
    """Custom callbacks that log per-episode RCSS metrics.

    Metrics surfaced to Ray Tune / TensorBoard:

    * ``episode_left_score``  — total goals scored by the left team.
    * ``episode_right_score`` — total goals scored by the right team.
    * ``episode_steps``       — number of environment steps in the episode.
    """

    def on_episode_end(
        self,
        *,
        worker: Any,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy],
        episode: EpisodeV2,
        **kwargs: Any,
    ) -> None:
        """Extract and record final scores from the episode's last info dict."""
        # Collect last-step info from any active agent.
        last_info: dict[str, Any] = {}
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info and "scores" in info:
                last_info = info
                break

        scores = last_info.get("scores", {"left": 0, "right": 0})
        episode.custom_metrics["episode_left_score"] = float(scores.get("left", 0))
        episode.custom_metrics["episode_right_score"] = float(scores.get("right", 0))
        episode.custom_metrics["episode_steps"] = float(
            last_info.get("step", episode.length)
        )

        logger.debug(
            "Episode finished | steps=%d | left=%d | right=%d",
            episode.length,
            scores.get("left", 0),
            scores.get("right", 0),
        )
