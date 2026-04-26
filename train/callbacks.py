from __future__ import annotations

import logging
from typing import Any, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

class RCSSCallbacks(DefaultCallbacks):
    """RLlib callbacks for RCSS training (new API stack).

    Logs per-episode metrics (scores and step count) via ``MetricsLogger``
    so that they appear in the Ray Tune result dict under the
    ``env_runners`` namespace.
    """

    def on_episode_end(
        self,
        *,
        episode: Any,
        env_runner: Optional[Any] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[Any] = None,
        env_index: int = 0,
        rl_module: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:

        last_info: dict[str, Any] = {}

        # New API stack: episode is always a MultiAgentEpisode.
        last_infos: dict[Any, Any] = episode.get_infos(-1) or {}
        for info in last_infos.values():
            if info and "scores" in info:
                last_info = info
                break
        episode_length = len(episode)

        scores = last_info.get("scores", {"left": 0, "right": 0})
        left_score = float(scores.get("left", 0))
        right_score = float(scores.get("right", 0))
        step = float(last_info.get("step", episode_length))

        if metrics_logger is not None:
            metrics_logger.log_value("episode_left_score", left_score, reduce="mean")
            metrics_logger.log_value("episode_right_score", right_score, reduce="mean")
            metrics_logger.log_value("episode_steps", step, reduce="mean")

        logger.debug(
            "Episode finished | steps=%d | left=%d | right=%d",
            episode_length,
            left_score,
            right_score,
        )
