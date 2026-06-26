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


def _coerce_image(image: Image | str) -> Image:
    return image if isinstance(image, Image) else Image(image=image)


def _fit_team_name(name: str, *, fallback: str) -> str:
    if not name or not name.isascii():
        name = fallback
    return name[:16]


def _default_team_name(config: GenDatasetCurriculumConfig, side: TeamSide, image: Image | str) -> str:
    normalized_image = _coerce_image(image)
    if side == TeamSide.LEFT and config.left_team_name_mapping is not None:
        return _fit_team_name(config.left_team_name_mapping(normalized_image), fallback="left-bots")
    if side == TeamSide.RIGHT and config.right_team_name_mapping is not None:
        return _fit_team_name(config.right_team_name_mapping(normalized_image), fallback="right-bots")
    return _fit_team_name(f"{normalized_image.name}-{side.value[0].upper()}", fallback=f"{side.value}-bots")


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
    resolved_left_team_name = left_team_name or _default_team_name(config, TeamSide.LEFT, left_image)
    resolved_right_team_name = right_team_name or _default_team_name(config, TeamSide.RIGHT, right_image)

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
            name=resolved_left_team_name,
            side=TeamSide.LEFT,
            players=make_bot_players(left_image),
            trainer=trainer_for(TeamSide.LEFT),
        ),
        right=TeamSchema(
            name=resolved_right_team_name,
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
