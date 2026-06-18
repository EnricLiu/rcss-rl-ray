from __future__ import annotations

from pydantic import IPvAnyAddress
from typing import Literal, cast

from client.base.allocator.config import AllocatorConfig
from rcss_env.config import EnvConfig
from schema.policy import DEFAULT_SSP_AGENT_IMAGE
from utils.config import ServerConfig

from .config import TrainConfig
from .curriculum import CurriculumMixin
from .curriculum.registry import build_curriculum_from_config, clone_curriculum_config
from .curriculum.dummy_marl import DummyMarlCurriculumConfig
from .curriculum.shooting import ShootingCurriculumConfig


def make_allocator_config(host: str, port: int) -> AllocatorConfig:
    return AllocatorConfig(base_url=f"http://{host}:{port}")


def make_server_config(host: str, port: int) -> ServerConfig:
    return ServerConfig(host=IPvAnyAddress(host), port=port)


def build_shooting_curriculum_config(train_cfg: TrainConfig) -> ShootingCurriculumConfig:
    if not isinstance(train_cfg.curriculum_config, ShootingCurriculumConfig):
        raise ValueError(f"Expected shooting curriculum, got {train_cfg.curriculum!r}")
    config = clone_curriculum_config(
        train_cfg.curriculum_config,
        grpc_server=make_server_config(train_cfg.grpc_host, train_cfg.grpc_port),
    )
    return cast(ShootingCurriculumConfig, config)


def build_dummy_marl_curriculum_config(train_cfg: TrainConfig) -> DummyMarlCurriculumConfig:
    if not isinstance(train_cfg.curriculum_config, DummyMarlCurriculumConfig):
        raise ValueError(f"Expected dummy_marl curriculum, got {train_cfg.curriculum!r}")
    config = clone_curriculum_config(
        train_cfg.curriculum_config,
        grpc_server=make_server_config(train_cfg.grpc_host, train_cfg.grpc_port),
    )
    return cast(DummyMarlCurriculumConfig, config)


def build_curriculum(train_cfg: TrainConfig) -> CurriculumMixin:
    return build_curriculum_from_config(
        train_cfg.curriculum_config,
        grpc_server=make_server_config(train_cfg.grpc_host, train_cfg.grpc_port),
    )


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
    player_agent_image: str = DEFAULT_SSP_AGENT_IMAGE,
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
