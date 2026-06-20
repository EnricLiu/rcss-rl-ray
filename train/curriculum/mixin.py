from abc import ABC, abstractmethod

from schema import GameServerSchema
from rcss_env.reward import RewardFnMixin


class CurriculumMixin(ABC):
    def agent_unums(self) -> tuple[int, ...]:
        """Return the stable uniform numbers controlled by learning policies.

        Concrete curricula should override this without sampling randomized room
        state. The schema-based fallback preserves compatibility for external
        curricula.
        """
        schema = self.make_schema()
        return tuple(
            sorted(player.unum for player in schema.teams.agent_team.ssp_agents())
        )

    @abstractmethod
    def make_schema(self) -> GameServerSchema:
        pass

    @abstractmethod
    def reward_fn(self) -> RewardFnMixin:
        pass
