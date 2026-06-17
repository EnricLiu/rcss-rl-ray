from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from utils.config import ServerConfig

from . import CurriculumMixin
from .dummy_marl import DummyMarlCurriculum, DummyMarlCurriculumConfig
from .shooting import ShootingCurriculum, ShootingCurriculumConfig

CurriculumConfigT = TypeVar(
    "CurriculumConfigT",
    ShootingCurriculumConfig,
    DummyMarlCurriculumConfig,
)


@dataclass(frozen=True)
class CurriculumRegistration:
    config_cls: type[ShootingCurriculumConfig] | type[DummyMarlCurriculumConfig]
    curriculum_cls: type[ShootingCurriculum] | type[DummyMarlCurriculum]


_REGISTRY: dict[str, CurriculumRegistration] = {
    "shooting": CurriculumRegistration(ShootingCurriculumConfig, ShootingCurriculum),
    "dummy_marl": CurriculumRegistration(DummyMarlCurriculumConfig, DummyMarlCurriculum),
}


def curriculum_names() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_curriculum_registration(name: str) -> CurriculumRegistration:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported curriculum: {name!r}") from exc


def clone_curriculum_config(
    config: ShootingCurriculumConfig | DummyMarlCurriculumConfig,
    *,
    grpc_server: ServerConfig,
) -> ShootingCurriculumConfig | DummyMarlCurriculumConfig:
    registration = get_curriculum_registration(config.type)
    payload = config.model_dump(mode="python", exclude={"grpc_server"})
    payload["grpc_server"] = grpc_server
    return registration.config_cls.model_validate(payload)


def build_curriculum_from_config(
    config: ShootingCurriculumConfig | DummyMarlCurriculumConfig,
    *,
    grpc_server: ServerConfig,
) -> CurriculumMixin:
    registration = get_curriculum_registration(config.type)
    curriculum_config = clone_curriculum_config(config, grpc_server=grpc_server)
    return registration.curriculum_cls(curriculum_config)

