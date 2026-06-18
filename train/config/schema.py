from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from train.curriculum.dummy_marl import DummyMarlCurriculumConfig
from train.curriculum.shooting import ShootingCurriculumConfig

TIMEZONE = timezone(timedelta(hours=+8))
DEFAULT_CHECKPOINT_SOURCE_METRIC = "env_runners/episode_return_mean"

CurriculumConfig = Annotated[
    ShootingCurriculumConfig | DummyMarlCurriculumConfig,
    Field(discriminator="type"),
]


class RuntimeConfig(BaseModel):
    algo: Literal["PPO"] = "PPO"
    ray_address: str | None = "auto"
    experiment_name: str = "rcss-shooting"
    storage_root: str = "/mnt/ray/storage"
    storage_path: str | None = None
    restore_path: str | None = None
    resume_from_checkpoint: str | None = None
    timestamp_experiment_name: bool = True
    num_samples: int = Field(default=1, ge=1)
    metric: str = "checkpoint_score"
    checkpoint_metric: str = "checkpoint_score"
    checkpoint_source_metric: str | None = None
    mode: Literal["min", "max"] = "max"
    log_to_file: bool = False
    num_iterations: int = Field(default=1000, ge=1)
    checkpoint_freq: int = Field(default=10, ge=0)
    checkpoint_num_to_keep: int | None = Field(default=20, ge=1)
    checkpoint_at_end: bool = True


class PPOTrainingConfig(BaseModel):
    num_env_runners: int = Field(default=4, ge=0)
    num_envs_per_runner: int = Field(default=1, ge=1)
    num_cpus_per_runner: float = Field(default=0.4, ge=0.0)
    num_learners: int = Field(default=0, ge=0)
    num_cpus_per_learner: float | Literal["auto"] = "auto"
    num_gpus_per_learner: float = Field(default=0.0, ge=0.0)
    train_batch_size: int = Field(default=4000, ge=1)
    sgd_minibatch_size: int = Field(default=128, ge=1)
    num_sgd_iter: int = Field(default=10, ge=1)
    lr: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)
    clip_param: float = Field(default=0.3, gt=0.0)


class GrpcConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=0, ge=0, le=65535)


class AllocatorConfig(BaseModel):
    host: str = "rcss-env-allocator.rcss-gateway-dev.svc.cluster.local"
    port: int = Field(default=80, ge=1, le=65535)


class InfrastructureConfig(BaseModel):
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    allocator: AllocatorConfig = Field(default_factory=AllocatorConfig)


class LoggingConfig(BaseModel):
    enable_aim: bool = True
    aim_repo: str = "/mnt/aim"
    aim_experiment_name: str | None = None
    aim_metrics: tuple[str, ...] | None = None


class TrainConfig(BaseModel):
    """Structured configuration for curriculum-based RLlib training."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ppo: PPOTrainingConfig = Field(default_factory=PPOTrainingConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    curriculum_config: CurriculumConfig = Field(
        default_factory=ShootingCurriculumConfig,
        validation_alias=AliasChoices("curriculum", "curriculum_config"),
        serialization_alias="curriculum",
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_legacy_config(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        cls._move_keys(
            data,
            "runtime",
            {
                "algo",
                "ray_address",
                "experiment_name",
                "storage_root",
                "storage_path",
                "restore_path",
                "resume_from_checkpoint",
                "timestamp_experiment_name",
                "num_samples",
                "metric",
                "checkpoint_metric",
                "checkpoint_source_metric",
                "mode",
                "log_to_file",
                "num_iterations",
                "checkpoint_freq",
                "checkpoint_num_to_keep",
                "checkpoint_at_end",
            },
        )
        cls._move_keys(
            data,
            "ppo",
            {
                "num_env_runners",
                "num_envs_per_runner",
                "num_cpus_per_runner",
                "num_learners",
                "num_cpus_per_learner",
                "num_gpus_per_learner",
                "train_batch_size",
                "sgd_minibatch_size",
                "num_sgd_iter",
                "lr",
                "gamma",
                "entropy_coeff",
                "clip_param",
            },
        )
        cls._move_infrastructure_keys(data)
        cls._move_keys(
            data,
            "logging",
            {
                "enable_aim",
                "aim_repo",
                "aim_experiment_name",
                "aim_metrics",
            },
        )
        cls._move_curriculum_keys(data)
        return data

    @staticmethod
    def _move_keys(data: dict[str, Any], section: str, keys: set[str]) -> None:
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            section_data = {} if section_data is None else {"value": section_data}

        for key in list(keys):
            if key in data:
                section_data[key] = data.pop(key)

        if section_data:
            data[section] = section_data

    @staticmethod
    def _move_infrastructure_keys(data: dict[str, Any]) -> None:
        infra = data.get("infrastructure")
        if not isinstance(infra, dict):
            infra = {}

        grpc = infra.get("grpc")
        if not isinstance(grpc, dict):
            grpc = {}
        allocator = infra.get("allocator")
        if not isinstance(allocator, dict):
            allocator = {}

        if "grpc_host" in data:
            grpc["host"] = data.pop("grpc_host")
        if "grpc_port" in data:
            grpc["port"] = data.pop("grpc_port")
        if "allocator_host" in data:
            allocator["host"] = data.pop("allocator_host")
        if "allocator_port" in data:
            allocator["port"] = data.pop("allocator_port")

        if grpc:
            infra["grpc"] = grpc
        if allocator:
            infra["allocator"] = allocator
        if infra:
            data["infrastructure"] = infra

    @staticmethod
    def _move_curriculum_keys(data: dict[str, Any]) -> None:
        curriculum = data.get("curriculum")
        if curriculum is None and "curriculum_config" in data:
            curriculum = data.get("curriculum_config")

        if isinstance(curriculum, str):
            curriculum_data: dict[str, Any] = {"type": curriculum}
        elif isinstance(curriculum, dict):
            curriculum_data = dict(curriculum)
        elif curriculum is None:
            curriculum_data = {}
        else:
            return

        mapping = {
            "curriculum_debug": "debug",
            "agent_unum": "agent_unum",
            "team_side": "team_side",
            "our_player_num": "our_player_num",
            "oppo_player_num": "oppo_player_num",
            "our_goalie_unum": "our_goalie_unum",
            "oppo_goalie_unum": "oppo_goalie_unum",
            "our_team_name": "our_team_name",
            "oppo_team_name": "oppo_team_name",
            "player_agent_image": "player_agent_image",
            "player_bot_image": "player_bot_image",
            "time_up": "time_up",
            "goal_l": "goal_l",
            "goal_r": "goal_r",
            "reward_goal": "reward_goal",
            "reward_concede": "reward_concede",
            "reward_out_of_bounds": "reward_out_of_bounds",
            "reward_kickable_bonus": "reward_kickable_bonus",
            "reward_agent_to_ball_shaping": "reward_agent_to_ball_shaping",
            "reward_ball_to_goal_shaping": "reward_ball_to_goal_shaping",
            "reward_ball_velocity_to_goal": "reward_ball_velocity_to_goal",
            "gamma_shaping": "gamma_shaping",
            "shaping_clip": "shaping_clip",
            "reward_time_decay": "reward_time_decay",
            "max_cycle_gap": "max_cycle_gap",
        }
        for flat_key, nested_key in mapping.items():
            if flat_key in data:
                curriculum_data[nested_key] = data.pop(flat_key)

        if curriculum_data:
            curriculum_data.setdefault("type", "shooting")
            data["curriculum"] = curriculum_data

    @model_validator(mode="after")
    def _finalize_defaults(self) -> TrainConfig:
        if self.runtime.restore_path is not None and self.runtime.resume_from_checkpoint is not None:
            raise ValueError("--restore and --resume-from-checkpoint are mutually exclusive")

        if self.runtime.resume_from_checkpoint is not None and self.runtime.num_samples != 1:
            raise ValueError("--resume-from-checkpoint only supports a single trial (num_samples=1)")

        if self.runtime.timestamp_experiment_name:
            time_suffix = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
            self.runtime.experiment_name = f"{self.runtime.experiment_name}-{time_suffix}"

        if self.runtime.checkpoint_source_metric is None and self.runtime.metric == self.runtime.checkpoint_metric:
            self.runtime.checkpoint_source_metric = DEFAULT_CHECKPOINT_SOURCE_METRIC
        elif self.runtime.checkpoint_source_metric is None:
            self.runtime.checkpoint_source_metric = self.runtime.metric

        if self.logging.enable_aim and self.logging.aim_experiment_name is None:
            self.logging.aim_experiment_name = self.runtime.experiment_name

        if self.runtime.storage_path is None:
            self.runtime.storage_path = self.runtime.storage_root

        return self

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "ray_address": self.ray_address,
            "experiment_name": self.experiment_name,
            "storage_root": self.storage_root,
            "storage_path": self.storage_path,
            "restore_path": self.restore_path,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "timestamp_experiment_name": self.timestamp_experiment_name,
            "num_samples": self.num_samples,
            "metric": self.metric,
            "checkpoint_metric": self.checkpoint_metric,
            "checkpoint_source_metric": self.checkpoint_source_metric,
            "mode": self.mode,
            "log_to_file": self.log_to_file,
            "num_env_runners": self.num_env_runners,
            "num_envs_per_runner": self.num_envs_per_runner,
            "num_cpus_per_runner": self.num_cpus_per_runner,
            "num_learners": self.num_learners,
            "num_cpus_per_learner": self.num_cpus_per_learner,
            "num_gpus_per_learner": self.num_gpus_per_learner,
            "train_batch_size": self.train_batch_size,
            "sgd_minibatch_size": self.sgd_minibatch_size,
            "num_sgd_iter": self.num_sgd_iter,
            "lr": self.lr,
            "gamma": self.gamma,
            "entropy_coeff": self.entropy_coeff,
            "clip_param": self.clip_param,
            "num_iterations": self.num_iterations,
            "checkpoint_freq": self.checkpoint_freq,
            "checkpoint_num_to_keep": self.checkpoint_num_to_keep,
            "checkpoint_at_end": self.checkpoint_at_end,
            "grpc_host": self.grpc_host,
            "grpc_port": self.grpc_port,
            "allocator_host": self.allocator_host,
            "allocator_port": self.allocator_port,
            "curriculum": self.curriculum,
            "curriculum_debug": self.curriculum_debug,
            "team_side": self.team_side,
            "our_goalie_unum": self.our_goalie_unum,
            "oppo_goalie_unum": self.oppo_goalie_unum,
            "our_team_name": self.our_team_name,
            "oppo_team_name": self.oppo_team_name,
            "player_agent_image": self.player_agent_image,
            "player_bot_image": self.player_bot_image,
            "time_up": self.time_up,
            "goal_l": self.goal_l,
            "goal_r": self.goal_r,
            "enable_aim": self.enable_aim,
            "aim_repo": self.aim_repo,
            "aim_experiment_name": self.aim_experiment_name,
            "aim_metrics": self.aim_metrics,
        } | self._shooting_legacy_dict()

    def _shooting_legacy_dict(self) -> dict[str, Any]:
        if not isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return {}
        cfg = self.curriculum_config
        return {
            "agent_unum": cfg.agent_unum,
            "our_player_num": cfg.our_player_num,
            "oppo_player_num": cfg.oppo_player_num,
            "reward_goal": cfg.reward_goal,
            "reward_concede": cfg.reward_concede,
            "reward_out_of_bounds": cfg.reward_out_of_bounds,
            "reward_kickable_bonus": cfg.reward_kickable_bonus,
            "reward_agent_to_ball_shaping": cfg.reward_agent_to_ball_shaping,
            "reward_ball_to_goal_shaping": cfg.reward_ball_to_goal_shaping,
            "reward_ball_velocity_to_goal": cfg.reward_ball_velocity_to_goal,
            "gamma_shaping": cfg.gamma_shaping,
            "shaping_clip": cfg.shaping_clip,
            "reward_time_decay": cfg.reward_time_decay,
            "max_cycle_gap": cfg.max_cycle_gap,
        }

    @property
    def algo(self) -> str:
        return self.runtime.algo

    @property
    def ray_address(self) -> str | None:
        return self.runtime.ray_address

    @property
    def experiment_name(self) -> str:
        return self.runtime.experiment_name

    @property
    def storage_root(self) -> str:
        return self.runtime.storage_root

    @property
    def storage_path(self) -> str | None:
        return self.runtime.storage_path

    @property
    def restore_path(self) -> str | None:
        return self.runtime.restore_path

    @property
    def resume_from_checkpoint(self) -> str | None:
        return self.runtime.resume_from_checkpoint

    @property
    def timestamp_experiment_name(self) -> bool:
        return self.runtime.timestamp_experiment_name

    @property
    def num_samples(self) -> int:
        return self.runtime.num_samples

    @property
    def metric(self) -> str:
        return self.runtime.metric

    @property
    def checkpoint_metric(self) -> str:
        return self.runtime.checkpoint_metric

    @property
    def checkpoint_source_metric(self) -> str | None:
        return self.runtime.checkpoint_source_metric

    @property
    def mode(self) -> str:
        return self.runtime.mode

    @property
    def log_to_file(self) -> bool:
        return self.runtime.log_to_file

    @property
    def num_iterations(self) -> int:
        return self.runtime.num_iterations

    @property
    def checkpoint_freq(self) -> int:
        return self.runtime.checkpoint_freq

    @property
    def checkpoint_num_to_keep(self) -> int | None:
        return self.runtime.checkpoint_num_to_keep

    @property
    def checkpoint_at_end(self) -> bool:
        return self.runtime.checkpoint_at_end

    @property
    def num_env_runners(self) -> int:
        return self.ppo.num_env_runners

    @property
    def num_envs_per_runner(self) -> int:
        return self.ppo.num_envs_per_runner

    @property
    def num_cpus_per_runner(self) -> float:
        return self.ppo.num_cpus_per_runner

    @property
    def num_learners(self) -> int:
        return self.ppo.num_learners

    @property
    def num_cpus_per_learner(self) -> float | str:
        return self.ppo.num_cpus_per_learner

    @property
    def num_gpus_per_learner(self) -> float:
        return self.ppo.num_gpus_per_learner

    @property
    def train_batch_size(self) -> int:
        return self.ppo.train_batch_size

    @property
    def sgd_minibatch_size(self) -> int:
        return self.ppo.sgd_minibatch_size

    @property
    def num_sgd_iter(self) -> int:
        return self.ppo.num_sgd_iter

    @property
    def lr(self) -> float:
        return self.ppo.lr

    @property
    def gamma(self) -> float:
        return self.ppo.gamma

    @property
    def entropy_coeff(self) -> float:
        return self.ppo.entropy_coeff

    @property
    def clip_param(self) -> float:
        return self.ppo.clip_param

    @property
    def grpc_host(self) -> str:
        return self.infrastructure.grpc.host

    @property
    def grpc_port(self) -> int:
        return self.infrastructure.grpc.port

    @property
    def allocator_host(self) -> str:
        return self.infrastructure.allocator.host

    @property
    def allocator_port(self) -> int:
        return self.infrastructure.allocator.port

    @property
    def curriculum(self) -> str:
        return self.curriculum_config.type

    @property
    def curriculum_debug(self) -> bool:
        return self.curriculum_config.debug

    @property
    def team_side(self) -> str:
        return self.curriculum_config.team_side

    @property
    def agent_unum(self) -> int:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.agent_unum
        return 1

    @property
    def our_player_num(self) -> int:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.our_player_num
        return 11

    @property
    def oppo_player_num(self) -> int:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.oppo_player_num
        return 11

    @property
    def our_goalie_unum(self) -> int | None:
        return self.curriculum_config.our_goalie_unum

    @property
    def oppo_goalie_unum(self) -> int | None:
        return self.curriculum_config.oppo_goalie_unum

    @property
    def our_team_name(self) -> str:
        return self.curriculum_config.our_team_name

    @property
    def oppo_team_name(self) -> str:
        return self.curriculum_config.oppo_team_name

    @property
    def player_agent_image(self) -> str:
        return self.curriculum_config.player_agent_image

    @property
    def player_bot_image(self) -> str:
        return self.curriculum_config.player_bot_image

    @property
    def time_up(self) -> int:
        return self.curriculum_config.time_up

    @property
    def goal_l(self) -> int | None:
        return self.curriculum_config.goal_l

    @property
    def goal_r(self) -> int | None:
        return self.curriculum_config.goal_r

    @property
    def reward_goal(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_goal
        return 0.0

    @property
    def reward_concede(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_concede
        return 0.0

    @property
    def reward_out_of_bounds(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_out_of_bounds
        return 0.0

    @property
    def reward_kickable_bonus(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_kickable_bonus
        return 0.0

    @property
    def reward_agent_to_ball_shaping(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_agent_to_ball_shaping
        return 0.0

    @property
    def reward_ball_to_goal_shaping(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_ball_to_goal_shaping
        return 0.0

    @property
    def reward_ball_velocity_to_goal(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_ball_velocity_to_goal
        return 0.0

    @property
    def gamma_shaping(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.gamma_shaping
        return self.gamma

    @property
    def shaping_clip(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.shaping_clip
        return 0.1

    @property
    def reward_time_decay(self) -> float:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.reward_time_decay
        return 0.0

    @property
    def max_cycle_gap(self) -> int:
        if isinstance(self.curriculum_config, ShootingCurriculumConfig):
            return self.curriculum_config.max_cycle_gap
        return 5

    @property
    def enable_aim(self) -> bool:
        return self.logging.enable_aim

    @property
    def aim_repo(self) -> str:
        return self.logging.aim_repo

    @property
    def aim_experiment_name(self) -> str | None:
        return self.logging.aim_experiment_name

    @property
    def aim_metrics(self) -> tuple[str, ...] | None:
        return self.logging.aim_metrics
