import random
from typing import override

from rcss_env.reward import DummyRewardFn, RewardFnMixin
from schema import (
    BotPolicy,
    GameServerSchema,
    PlayerInitState,
    PlayerSchema,
    Policy,
    Position,
    RefereeSchema,
    RoomInitState,
    StoppingEvents,
    TeamSchema,
    TeamSide,
    TeamsSchema,
)
from schema.coach import CoachSchema

from ..mixin import CurriculumMixin
from .config import DummyMarlCurriculumConfig


class DummyMarlCurriculum(CurriculumMixin):
    """Minimal 11-vs-11 MARL curriculum for wiring and training smoke tests."""

    def __init__(self, config: DummyMarlCurriculumConfig):
        self.config = config
        self.__reward = DummyRewardFn()

    @override
    def make_schema(self) -> GameServerSchema:
        return GameServerSchema(
            teams=self.__make_teams(),
            stopping=self.__make_stopping_events(),
            referee=RefereeSchema(enable=True),
            init_state=RoomInitState(ball=Position(x=0.0, y=0.0)),
            log=self.config.debug,
        )

    @override
    def reward_fn(self) -> RewardFnMixin:
        return self.__reward

    def __make_teams(self) -> TeamsSchema:
        is_agent_left = is_agent_right = False
        match self.config.team_side.lower():
            case "left":
                is_agent_left = True
            case "right":
                is_agent_right = True
            case "rand":
                is_agent_left = random.random() > 0.5
                is_agent_right = not is_agent_left

        return TeamsSchema(
            left=self.__make_team(TeamSide.LEFT, is_agent_left),
            right=self.__make_team(TeamSide.RIGHT, is_agent_right),
        )

    def __make_team(self, side: TeamSide, is_agent: bool) -> TeamSchema:
        goalie_unum = self.config.our_goalie_unum if is_agent else self.config.oppo_goalie_unum
        players = [
            self.__make_player(
                unum=unum,
                side=side,
                is_agent=is_agent,
                is_goalie=(goalie_unum is not None and unum == goalie_unum),
            )
            for unum in range(1, 12)
        ]

        coach = None
        if is_agent:
            coach = CoachSchema(
                policy=Policy.ssp_agent(
                    image=self.config.player_agent_image,
                    grpc_host=self.config.grpc_server.host,
                    grpc_port=self.config.grpc_server.port,
                ),
            )

        return TeamSchema(
            name=self.config.our_team_name if is_agent else self.config.oppo_team_name,
            side=side,
            players=players,
            coach=coach,
        )

    def __make_player(
        self,
        *,
        unum: int,
        side: TeamSide,
        is_agent: bool,
        is_goalie: bool,
    ) -> PlayerSchema:
        if is_agent:
            policy = Policy.ssp_agent(
                image=self.config.player_agent_image,
                grpc_host=self.config.grpc_server.host,
                grpc_port=self.config.grpc_server.port,
            )
        else:
            policy = BotPolicy(image=self.config.player_bot_image)

        return PlayerSchema(
            unum=unum,
            goalie=is_goalie,
            policy=policy,
            init_state=PlayerInitState(
                pos=self.__formation_position(unum=unum, side=side),
                stamina=6000,
            ),
        )

    def __formation_position(self, *, unum: int, side: TeamSide) -> Position:
        base = {
            1: (-50.0, 0.0),
            2: (-36.0, -22.0),
            3: (-38.0, -8.0),
            4: (-38.0, 8.0),
            5: (-36.0, 22.0),
            6: (-20.0, -18.0),
            7: (-18.0, -6.0),
            8: (-18.0, 6.0),
            9: (-20.0, 18.0),
            10: (-4.0, -10.0),
            11: (-4.0, 10.0),
        }
        x, y = base[unum]
        if side == TeamSide.RIGHT:
            x = -x
            y = -y
        return Position(x=x, y=y)

    def __make_stopping_events(self) -> StoppingEvents:
        return StoppingEvents(
            time_up=self.config.time_up,
            goal_l=self.config.goal_l,
            goal_r=self.config.goal_r,
        )
