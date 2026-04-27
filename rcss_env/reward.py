"""Reward computation.

Computes a per-step reward from consecutive WorldModel frames.
Current implementation: own goal scored +1, opponent goal scored -1.
"""

from abc import ABC, abstractmethod
from typing import Mapping
from typing import override

from .grpc_srv.proto import pb2


class RewardFnMixin(ABC):
    def __init__(self) -> None:
        self._last_reward_breakdown: dict[str, float] = {}

    @property
    def last_reward_breakdown(self) -> dict[str, float]:
        return self._last_reward_breakdown.copy()

    def reset_reward_breakdown(self) -> None:
        self._last_reward_breakdown = {}

    def set_reward_breakdown(self, breakdown: Mapping[str, float] | None) -> None:
        if breakdown is None:
            self.reset_reward_breakdown()
            return
        self._last_reward_breakdown = {
            key: float(value)
            for key, value in breakdown.items()
        }

    @abstractmethod
    def compute(
        self,
        prev_obs: pb2.WorldModel | None,
        prev_truth: pb2.WorldModel | None,
        curr_obs: pb2.WorldModel,
        curr_truth: pb2.WorldModel
    ): pass


class DummyRewardFn(RewardFnMixin):
    def __init__(self) -> None:
        super().__init__()

    @override
    def compute(
        self,
        prev_obs: pb2.WorldModel | None,
        prev_truth: pb2.WorldModel | None,
        curr_obs: pb2.WorldModel,
        curr_truth: pb2.WorldModel,
    ) -> float:
        """Compute the reward from previous and current observation / ground-truth world models.

        Args:
            prev_obs: Observation world model from the previous frame (None on the first step).
            prev_truth: Full-information world model from the previous frame (None on the first step).
            curr_obs: Observation world model for the current frame.
            curr_truth: Full-information world model for the current frame.

        Returns:
            Scalar reward value.
        """
        if prev_obs is None:
            self.reset_reward_breakdown()
            return 0.0

        rewards = 0.0

        prev_score_wm = prev_truth or prev_obs

        # Goal-difference reward: own goal +1, opponent goal -1
        score_diff = curr_truth.our_team_score - prev_score_wm.our_team_score
        opp_diff = curr_truth.their_team_score - prev_score_wm.their_team_score
        rewards += float(score_diff)
        rewards -= float(opp_diff)

        self.set_reward_breakdown({
            "goal": float(score_diff),
            "concede": -float(opp_diff),
        })

        return rewards


def distance(pos1: pb2.Vector2D, pos2: pb2.Vector2D) -> float:
    """Compute the Euclidean distance between two Vector2D points."""
    return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5
