"""Inference runtime configuration without model-topology overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from rcss_env.config import EnvConfig
from train.config.loader import load_config_mapping
from train.config.schema import (
    AllocatorConfig,
    CurriculumConfig,
    GrpcConfig,
    InfrastructureConfig,
)
from train.curriculum.registry import build_curriculum_from_config
from train.factory import make_allocator_config, make_server_config


class InferenceLoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    trace_actions: bool = False


class InferenceConfig(BaseModel):
    """Validated runtime inputs. Policy topology is intentionally absent."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    bundle_path: str | Path = Field(
        validation_alias=AliasChoices("bundle", "bundle_path")
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool | None = None
    fallback_on_policy_error: bool = True
    episodes: int = Field(default=1, ge=1)
    seed: int = 0
    max_episode_retries: int = Field(default=0, ge=0)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    curriculum_config: CurriculumConfig = Field(
        validation_alias=AliasChoices("curriculum", "curriculum_config")
    )
    logging: InferenceLoggingConfig = Field(default_factory=InferenceLoggingConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_policy_mapping_overrides(cls, value: Any) -> Any:
        if isinstance(value, dict):
            forbidden = {"agent_to_module", "policy_mapping", "module_ids"}
            present = forbidden.intersection(value)
            if present:
                raise ValueError(
                    "Inference config cannot override manifest policy topology: "
                    + ", ".join(sorted(present))
                )
        return value

    def build_env_config(self) -> EnvConfig:
        grpc = make_server_config(
            self.infrastructure.grpc.host,
            self.infrastructure.grpc.port,
        )
        curriculum = build_curriculum_from_config(
            self.curriculum_config,
            grpc_server=grpc,
        )
        return EnvConfig(
            grpc=grpc,
            allocator=make_allocator_config(
                self.infrastructure.allocator.host,
                self.infrastructure.allocator.port,
            ),
            curriculum=curriculum,
        )


def load_inference_config(path: str | Path) -> InferenceConfig:
    return InferenceConfig.model_validate(load_config_mapping(path))


__all__ = [
    "AllocatorConfig",
    "GrpcConfig",
    "InferenceConfig",
    "InferenceLoggingConfig",
    "load_inference_config",
]
