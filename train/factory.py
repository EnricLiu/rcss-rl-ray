from __future__ import annotations

from typing import Literal

from client.base.allocator.config import AllocatorConfig
from rcss_env.config import EnvConfig
from utils.config import ServerConfig

from .config import TrainConfig
from .curriculum import CurriculumMixin
from .curriculum.shooting import ShootingCurriculum, ShootingCurriculumConfig


def make_allocator_config(host: str, port: int) -> AllocatorConfig:
    return AllocatorConfig(base_url=f"http://{host}:{port}")


def make_server_config(host: str, port: int) -> ServerConfig:
    return ServerConfig(host=host, port=port)


def build_shooting_curriculum_config(train_cfg: TrainConfig) -> ShootingCurriculumConfig:
    return ShootingCurriculumConfig(
        debug=train_cfg.curriculum_debug,
        agent_unum=train_cfg.agent_unum,
        team_side=train_cfg.team_side,
        grpc_server=make_server_config(train_cfg.grpc_host, train_cfg.grpc_port),
        our_player_num=train_cfg.our_player_num,
        oppo_player_num=train_cfg.oppo_player_num,
        our_goalie_unum=train_cfg.our_goalie_unum,
        oppo_goalie_unum=train_cfg.oppo_goalie_unum,
        time_up=train_cfg.time_up,
        goal_l=train_cfg.goal_l,
        goal_r=train_cfg.goal_r,
        our_team_name=train_cfg.our_team_name,
        oppo_team_name=train_cfg.oppo_team_name,
        player_agent_image=train_cfg.player_agent_image,
        player_bot_image=train_cfg.player_bot_image,
    )


def build_curriculum(train_cfg: TrainConfig) -> CurriculumMixin:
    match train_cfg.curriculum:
        case "shooting":
            return ShootingCurriculum(build_shooting_curriculum_config(train_cfg))
        case _:
            raise ValueError(f"Unsupported curriculum: {train_cfg.curriculum!r}")


def build_env_config(train_cfg: TrainConfig) -> EnvConfig:
    return EnvConfig(
        grpc=make_server_config(train_cfg.grpc_host, train_cfg.grpc_port),
        allocator=make_allocator_config(train_cfg.allocator_host, train_cfg.allocator_port),
        curriculum=build_curriculum(train_cfg),
    )


def make_shooting_env_config(
    *,
    grpc_host: str,
    grpc_port: int,
    allocator_host: str,
    allocator_port: int,
    our_player_num: int,
    oppo_player_num: int | None = None,
    agent_unum: int = 1,
    team_side: Literal["left", "right", "rand"] = "left",
    time_up: int = 5000,
    player_agent_image: str = "Cyrus2D/SoccerSimulationProxy",
    player_bot_image: str = "HELIOS/helios-base",
    our_goalie_unum: int | None = 1,
    oppo_goalie_unum: int | None = 1,
) -> EnvConfig:
    player_count = int(our_player_num)
    train_cfg = TrainConfig(
        grpc_host=str(grpc_host),
        grpc_port=grpc_port,
        allocator_host=allocator_host,
        allocator_port=allocator_port,
        agent_unum=agent_unum,
        team_side=team_side,
        our_player_num=player_count,
        oppo_player_num=player_count if oppo_player_num is None else int(oppo_player_num),
        our_goalie_unum=our_goalie_unum,
        oppo_goalie_unum=oppo_goalie_unum,
        time_up=time_up,
        player_agent_image=player_agent_image,
        player_bot_image=player_bot_image,
    )
    return build_env_config(train_cfg)
