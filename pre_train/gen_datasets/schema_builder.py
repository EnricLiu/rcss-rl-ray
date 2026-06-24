from __future__ import annotations

from ipaddress import IPv4Address
from typing import Iterable

from pydantic import IPvAnyAddress

from schema import (
    BotPolicy,
    GameServerSchema,
    PlayerSchema,
    Policy,
    RefereeSchema,
    StoppingEvents,
    TeamSchema,
    TeamsSchema,
    TeamSide,
    TrainerSchema,
)

from .config import GenDatasetCurriculumConfig, Image


def make_bot_players(image: Image | str, *, unums: Iterable[int] = range(1, 12)) -> list[PlayerSchema]:
    image_name = image.image if isinstance(image, Image) else image
    return [
        PlayerSchema(
            unum=unum,
            goalie=(unum == 1),
            policy=BotPolicy(image=image_name),
        )
        for unum in unums
    ]


def build_pretrain_schema(
    config: GenDatasetCurriculumConfig,
    *,
    left_image: Image | str,
    right_image: Image | str,
    left_team_name: str | None = None,
    right_team_name: str | None = None,
    grpc_host: IPvAnyAddress | IPv4Address,
    grpc_port: int,
) -> GameServerSchema:
    trainer_sides = set(config.trainer.sides)

    def trainer_for(side: TeamSide) -> TrainerSchema | None:
        if side not in trainer_sides:
            return None
        return TrainerSchema(
            policy=Policy.ssp_agent(
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                image=config.trainer.image,
            )
        )

    teams = TeamsSchema(
        left=TeamSchema(
            name=left_team_name,
            side=TeamSide.LEFT,
            players=make_bot_players(left_image),
            trainer=trainer_for(TeamSide.LEFT),
        ),
        right=TeamSchema(
            name=right_team_name,
            side=TeamSide.RIGHT,
            players=make_bot_players(right_image),
            trainer=trainer_for(TeamSide.RIGHT),
        ),
    )

    return GameServerSchema(
        teams=teams,
        stopping=StoppingEvents(time_up=config.time_up),
        referee=RefereeSchema(enable=True),
        log=config.log,
    )
