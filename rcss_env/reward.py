"""Reward computation.

Computes a per-step reward from consecutive WorldModel frames.
Current implementation: own goal scored +1, opponent goal scored -1.
"""

from abc import ABC, abstractmethod
from typing import override

from .grpc_srv.proto import pb2

class RewardFnMixin(ABC):
    @abstractmethod
    def compute(
        self,
        prev_obs: pb2.WorldModel | None,
        prev_truth: pb2.WorldModel | None,
        curr_obs: pb2.WorldModel,
        curr_truth: pb2.WorldModel
    ): pass

class DummyRewardFn(RewardFnMixin):
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
        rewards = 0.0

        # Goal-difference reward: own goal +1, opponent goal -1
        score_diff = curr_obs.our_team_score - prev_obs.our_team_score
        opp_diff = curr_obs.their_team_score - prev_obs.their_team_score
        rewards += float(score_diff)
        rewards -= float(opp_diff)

        ball_to_goal_distance = curr_obs.ball.position

        return rewards


def distance(pos1: pb2.Vector2D, pos2: pb2.Vector2D) -> float:
    """Compute the Euclidean distance between two Vector2D points."""
    return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5
