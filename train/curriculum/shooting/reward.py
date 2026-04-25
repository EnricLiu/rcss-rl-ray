from typing import override

from rcss_env.grpc_srv.proto import pb2
from rcss_env.reward import RewardFnMixin

from .config import ShootingCurriculumConfig

class ShootingReward(RewardFnMixin):
    def __init__(self, config: ShootingCurriculumConfig):
        self.config = config

    @override
    def compute(
        self,
        prev_obs: pb2.WorldModel | None,
        prev_truth: pb2.WorldModel | None,
        curr_obs: pb2.WorldModel,
        curr_truth: pb2.WorldModel
    ):
        pass