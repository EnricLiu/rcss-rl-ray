"""Training and environment configuration dataclasses."""

from __future__ import annotations
from dataclasses import dataclass, field

from .env import EnvConfig
from .train import TrainConfig


@dataclass
class Config:
    env: EnvConfig
    train: TrainConfig = field(default_factory=TrainConfig)

