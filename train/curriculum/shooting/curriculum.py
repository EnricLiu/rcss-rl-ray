import random
from typing import override

from rcss_env.reward import RewardFnMixin
from schema import GameServerSchema, TeamSchema, TeamsSchema, StoppingEvents, RefereeSchema, RoomInitState, Position, \
    TeamSide, PlayerSchema, PlayerInitState, SspAgentPolicy, BotPolicy
from schema.coach import CoachSchema
from ..mixin import CurriculumMixin

from .reward import ShootingReward
from .config import ShootingCurriculumConfig


class ShootingCurriculum(CurriculumMixin):
    def __init__(self, config: ShootingCurriculumConfig):
        self.config = config
        self.__reward = ShootingReward(config)

    @override
    def make_schema(self) -> GameServerSchema:
        ret = GameServerSchema(
            teams=self.__make_teams(),
            stopping=self.__make_stopping_events(),
            referee=self.__make_referee(),
            init_state=self.__make_init_state(),
            log=self.config.debug,
        )
        return ret

    @override
    def reward_fn(self) -> RewardFnMixin:
        return self.__reward

    def __make_team(self, side: TeamSide, is_agent: bool) -> TeamSchema:
        name = self.config.our_team_name if is_agent else self.config.oppo_team_name

        player_num = self.config.our_player_num if is_agent else self.config.oppo_player_num
        goalie_unum = self.config.our_goalie_unum if is_agent else self.config.oppo_goalie_unum

        unum_iter = range(1, player_num+1)
        if is_agent: unum_iter = filter(lambda u: u != self.config.agent_unum, unum_iter)

        players = [self.__make_bot_player(unum, unum == goalie_unum) for unum in unum_iter]

        if is_agent:
            players.append(self.__make_agent_player())

        coach = None
        if is_agent:
            coach = CoachSchema(
                policy=SspAgentPolicy(
                    image=self.config.player_agent_image,
                    grpc_host=self.config.grpc_server.host,
                    grpc_port=self.config.grpc_server.port,
                ),
            )

        ret = TeamSchema(
            name=name,
            side=side,
            players=players,
            coach=coach,
        )

        return ret

    def __make_agent_player(self) -> PlayerSchema:
        unum = self.config.agent_unum
        is_goalie = (self.config.our_goalie_unum is not None) and (unum == self.config.our_goalie_unum)
        ret = PlayerSchema(
            unum=unum,
            goalie=is_goalie,
            policy=SspAgentPolicy(
                image=self.config.player_agent_image,
                grpc_host=self.config.grpc_server.host,
                grpc_port=self.config.grpc_server.port,
            ),
            init_state=self.__make_agent_player_init_state(),
            # blocklist=None
        )

        return ret

    def __make_agent_player_init_state(self) -> PlayerInitState:
        ret = PlayerInitState(
            pos=self.config.agent_region.sample_p99(),
            stamina=6000,
        )

        return ret

    def __make_bot_player(self, unum: int, goalie: bool) -> PlayerSchema:
        ret = PlayerSchema(
            unum=unum,
            goalie=goalie,
            policy=BotPolicy(
                image=self.config.player_bot_image
            ),
        )

        return ret

    def __make_init_ball_pos(self) -> Position:
        return self.config.ball_region.sample_p99()

    def __make_teams(self) -> TeamsSchema:
        is_agent_left = is_agent_right = False
        match self.config.team_side.lower():
            case "left": is_agent_left = True
            case "right": is_agent_right = True
            case "rand":
                rand = random.random()
                is_agent_left = rand > 0.5
                is_agent_right = not is_agent_left

        ret = TeamsSchema(
            left=self.__make_team(TeamSide.LEFT, is_agent_left),
            right=self.__make_team(TeamSide.RIGHT, is_agent_right),
        )
        return ret

    def __make_init_state(self) -> RoomInitState:
        ret = RoomInitState(
            ball=self.__make_init_ball_pos(),
        )

        return ret

    def __make_referee(self) -> RefereeSchema:
        ret = RefereeSchema(
            enable=True,
        )

        return ret

    def __make_stopping_events(self) -> StoppingEvents:
        ret = StoppingEvents(
            time_up=self.config.time_up,
            goal_l=self.config.goal_l,
            goal_r=self.config.goal_r,
        )

        return ret
