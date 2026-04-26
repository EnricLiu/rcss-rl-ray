from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

TIMEZONE = timezone(timedelta(hours=+8))

@dataclass
class TrainConfig:
    """Configuration for curriculum-based RLlib training launched through Tune."""

    # Ray / Tune runtime
    algo: Literal["PPO"] = "PPO"
    ray_address: str | None = "auto"
    experiment_name: str = "rcss-shooting"
    storage_root: str = "/mnt/ray/storage"
    storage_path: str | None = None
    restore_path: str | None = None
    timestamp_experiment_name: bool = True
    num_samples: int = Field(default=1, ge=1)
    metric: str = "env_runners/episode_reward_mean"
    mode: Literal["min", "max"] = "max"
    log_to_file: bool = False

    # PPO / RLlib hyperparameters
    num_env_runners: int = Field(default=4, ge=0)
    num_envs_per_runner: int = Field(default=1, ge=1)
    train_batch_size: int = Field(default=4000, ge=1)
    sgd_minibatch_size: int = Field(default=128, ge=1)
    num_sgd_iter: int = Field(default=10, ge=1)
    lr: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)
    clip_param: float = Field(default=0.3, gt=0.0)
    num_iterations: int = Field(default=100, ge=1)
    checkpoint_freq: int = Field(default=10, ge=0)
    checkpoint_num_to_keep: int | None = Field(default=3, ge=1)
    checkpoint_at_end: bool = True

    # Infrastructure
    grpc_host: str = "0.0.0.0"
    grpc_port: int = Field(default=0, ge=0, le=65535)
    allocator_host: str = "rcss-env-allocator.rcss-gateway-dev.svc.cluster.local"
    allocator_port: int = Field(default=80, ge=1, le=65535)

    # Curriculum selection and shooting curriculum parameters
    curriculum: Literal["shooting"] = "shooting"
    curriculum_debug: bool = True
    agent_unum: int = Field(default=2, ge=1, le=11)
    team_side: Literal["left", "right", "rand"] = "left"
    our_player_num: int = Field(default=2, ge=1, le=11)
    oppo_player_num: int = Field(default=1, ge=1, le=11)
    our_goalie_unum: int | None = Field(default=1, ge=1, le=11)
    oppo_goalie_unum: int | None = Field(default=1, ge=1, le=11)
    our_team_name: str = "nexus-prime"
    oppo_team_name: str = "bot"
    player_agent_image: str = "Cyrus2D/SoccerSimulationProxy"
    player_bot_image: str = "HELIOS/helios-base"
    time_up: int = Field(default=5000, ge=0, le=65535)
    goal_l: int | None = Field(default=1, ge=0, le=255)
    goal_r: int | None = Field(default=1, ge=0, le=255)
    reward_goal: float = Field(default=10.0, ge=0.0)
    reward_concede: float = Field(default=10.0, ge=0.0)
    reward_out_of_bounds: float = Field(default=1.0, ge=0.0)
    reward_ball_to_goal_shaping: float = Field(default=1.0, ge=0.0)
    reward_time_decay: float = Field(default=0.001, ge=0.0)

    # Aim/Tune logging
    enable_aim: bool = True
    aim_repo: str = "/mnt/aim"
    aim_experiment_name: str | None = None
    aim_metrics: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.timestamp_experiment_name:
            time_suffix = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.experiment_name}-{time_suffix}"

        if self.enable_aim and self.aim_experiment_name is None:
            self.aim_experiment_name = self.experiment_name

        if self.storage_path is None:
            self.storage_path = self.storage_root
