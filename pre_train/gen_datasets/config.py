from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from client.base.allocator import AllocatorConfig
from schema import DEFAULT_SSP_AGENT_IMAGE, TeamSide
from utils.config import ServerConfig


class ImageKind(str, Enum):
    SSP = "ssp"
    HELIOS = "helios"


class SaveMode(str, Enum):
    OBS = "obs"
    STATE = "state"
    BOTH = "both"


class Image(BaseModel):
    model_config = ConfigDict(frozen=True)

    image: str

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("image must use provider/name format")
        return value

    @property
    def provider(self) -> str:
        return self.image.split("/", maxsplit=1)[0]

    @property
    def name(self) -> str:
        return self.image.split("/", maxsplit=1)[1]


class ImagesConfig(BaseModel):
    type: ImageKind
    images: dict[str, list[Image]]

    @field_validator("images", mode="before")
    @classmethod
    def _parse_images(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            provider: [
                item if isinstance(item, (dict, Image)) else {"image": str(item)}
                for item in images
            ]
            for provider, images in value.items()
        }


def default_left_team_name(image: Image) -> str:
    return f"{image.name}-L"


def default_right_team_name(image: Image) -> str:
    return f"{image.name}-R"


class TrainerDatasetConfig(BaseModel):
    sides: list[TeamSide] = Field(default_factory=lambda: [TeamSide.LEFT])
    image: str = DEFAULT_SSP_AGENT_IMAGE


class RayDatasetExecutionConfig(BaseModel):
    enabled: bool = False
    max_concurrent_matches: int = Field(default=1, ge=1)
    num_cpus_per_match: float = Field(default=1.0, gt=0)
    num_gpus_per_match: float = Field(default=0.0, ge=0)
    task_max_retries: int = Field(default=0, ge=0)
    scheduling_strategy: str = "default"
    summary_filename: str = "batch_summary.json"

    @field_validator("summary_filename")
    @classmethod
    def _validate_summary_filename(cls, value: str) -> str:
        if not value or Path(value).name != value:
            raise ValueError("summary_filename must be a file name without directories")
        return value


class GenDatasetCurriculumConfig(BaseModel):
    type: Literal["gen_dataset"] = "gen_dataset"
    debug: bool = False

    output_root: Path = Path("datasets/pretrain")
    dataset_name: str = "rcss-pretrain"
    save_mode: SaveMode = SaveMode.OBS

    image_pool: list[Image] = Field(default_factory=lambda: [Image(image="HELIOS/helios-base")])
    left_team_name_mapping: Callable[[Image], str] | None = Field(default=default_left_team_name, exclude=True)
    right_team_name_mapping: Callable[[Image], str] | None = Field(default=default_right_team_name, exclude=True)

    allocator: AllocatorConfig = Field(default_factory=lambda: AllocatorConfig(base_url="http://127.0.0.1:5555"))
    grpc_server: ServerConfig = Field(default_factory=ServerConfig)
    trainer: TrainerDatasetConfig = Field(default_factory=TrainerDatasetConfig)
    ray: RayDatasetExecutionConfig = Field(default_factory=RayDatasetExecutionConfig)

    time_up: int = Field(default=5000, ge=0, le=65535)
    matches: int = Field(default=1, ge=1)
    random_seed: int | None = None
    log: bool = True

    @field_validator("image_pool", mode="before")
    @classmethod
    def _parse_image_pool(cls, value: Any) -> Any:
        if isinstance(value, set):
            value = list(value)
        if isinstance(value, list):
            return [
                item if isinstance(item, (dict, Image)) else {"image": str(item)}
                for item in value
            ]
        return value

    @model_validator(mode="after")
    def _validate_ray_capacity(self) -> GenDatasetCurriculumConfig:
        if self.ray.max_concurrent_matches > self.matches:
            raise ValueError("ray.max_concurrent_matches must not exceed matches")
        return self
