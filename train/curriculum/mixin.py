from abc import ABC, abstractmethod

from schema import GameServerSchema
from rcss_env.reward import RewardFnMixin


class CurriculumMixin(ABC):
    @abstractmethod
    def make_schema(self) -> GameServerSchema:
        pass

    @abstractmethod
    def reward_fn(self) -> RewardFnMixin:
        pass