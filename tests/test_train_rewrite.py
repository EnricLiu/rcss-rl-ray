from __future__ import annotations

from typing import Any, cast

import pytest

from train.callbacks import RCSSCallbacks
from train.factory import build_env_config
from train.train import (
    ENV_NAME,
    build_callbacks_class,
    build_ppo_config,
    build_run_config,
    build_train_config,
    build_tune_callbacks,
    build_tune_config,
    parse_args,
)
from train.curriculum.shooting import ShootingCurriculum


class _FakeEpisode:
    def __init__(self) -> None:
        self._infos: dict[int, dict[str, Any]] = {}
        self.user_data: dict[str, Any] = {}
        self.custom_metrics: dict[str, float] = {}
        self.length = 0

    def get_agents(self) -> list[int]:
        return list(self._infos.keys())

    def last_info_for(self, agent_id: int) -> dict[str, Any] | None:
        return self._infos.get(agent_id)


def test_build_train_config_parses_tune_and_aim_flags() -> None:
    args = parse_args(
        [
            "--ray-address",
            "local",
            "--disable-aim",
            "--aim-metrics",
            "env_runners/episode_reward_mean,custom_metrics/episode_steps_mean",
            "--num-iterations",
            "7",
            "--our-goalie-unum",
            "none",
            "--goal-r",
            "null",
        ]
    )

    cfg = build_train_config(args)

    assert cfg.ray_address == "local"
    assert cfg.enable_aim is False
    assert cfg.aim_metrics == (
        "env_runners/episode_reward_mean",
        "custom_metrics/episode_steps_mean",
    )
    assert cfg.num_iterations == 7
    assert cfg.our_goalie_unum is None
    assert cfg.goal_r is None


def test_default_aim_experiment_tracks_train_experiment_name() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--no-timestamp-experiment-name",
                "--experiment-name",
                "unit-train",
            ]
        )
    )

    assert cfg.enable_aim is True
    assert cfg.experiment_name == "unit-train"
    assert cfg.aim_experiment_name == "unit-train"


def test_build_env_config_uses_shooting_curriculum() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--disable-aim",
                "--grpc-host",
                "127.0.0.1",
                "--grpc-port",
                "43123",
                "--allocator-host",
                "allocator.default.svc",
                "--allocator-port",
                "8080",
                "--our-player-num",
                "3",
                "--oppo-player-num",
                "2",
                "--time-up",
                "321",
                "--reward-goal",
                "12.5",
                "--reward-kickable-bonus",
                "0.7",
                "--reward-agent-to-ball-shaping",
                "1.8",
                "--reward-time-decay",
                "0.02",
                "--reward-ball-velocity-to-goal",
                "0.12",
                "--gamma-shaping",
                "0.97",
                "--shaping-clip",
                "0.08",
                "--max-cycle-gap",
                "7",
            ]
        )
    )

    env_cfg = build_env_config(cfg)
    schema = env_cfg.curriculum.make_schema()
    opponent_team = schema.teams.right if schema.teams.agent_team.side == schema.teams.left.side else schema.teams.left

    assert env_cfg.grpc.addr == "127.0.0.1:43123"
    assert env_cfg.allocator.base_url == "http://allocator.default.svc:8080"
    assert isinstance(env_cfg.curriculum, ShootingCurriculum)
    assert env_cfg.curriculum.config.reward_goal == 12.5
    assert env_cfg.curriculum.config.reward_kickable_bonus == 0.7
    assert env_cfg.curriculum.config.reward_agent_to_ball_shaping == 1.8
    assert env_cfg.curriculum.config.reward_ball_velocity_to_goal == 0.12
    assert env_cfg.curriculum.config.gamma_shaping == 0.97
    assert env_cfg.curriculum.config.shaping_clip == 0.08
    assert env_cfg.curriculum.config.reward_time_decay == 0.02
    assert env_cfg.curriculum.config.max_cycle_gap == 7
    assert schema.stopping.time_up == 321
    assert len(schema.teams.agent_team.players) == 3
    assert len(opponent_team.players) == 2


def test_build_ppo_config_keeps_registered_env_and_legacy_model_stack() -> None:
    cfg = build_train_config(parse_args(["--ray-address", "local", "--disable-aim"]))
    env_cfg = build_env_config(cfg)

    ppo_config = build_ppo_config(cfg, env_cfg)

    assert ppo_config.env == ENV_NAME
    assert ppo_config.env_config == {"env_config": env_cfg}
    assert ppo_config.disable_env_checking is True
    assert ppo_config.enable_rl_module_and_learner is False
    assert ppo_config.enable_env_runner_and_connector_v2 is False
    assert ppo_config.model["custom_model"] == "rcss_fcnet"
    assert issubclass(ppo_config.callbacks_class, RCSSCallbacks)
    assert ppo_config.callbacks_class.CHECKPOINT_SCORE_ATTRIBUTE == cfg.checkpoint_metric
    assert ppo_config.callbacks_class.CHECKPOINT_SCORE_SOURCE_ATTRIBUTE == cfg.checkpoint_source_metric


def test_build_tune_config_and_run_config_without_aim() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--disable-aim",
                "--no-timestamp-experiment-name",
                "--experiment-name",
                "unit-train",
                "--num-samples",
                "2",
                "--checkpoint-freq",
                "0",
            ]
        )
    )

    assert build_tune_callbacks(cfg) == []

    tune_config = build_tune_config(cfg)
    run_config = build_run_config(cfg)

    assert tune_config.num_samples == 2
    assert tune_config.metric == cfg.metric
    assert run_config.name == "unit-train"
    assert run_config.stop == {"training_iteration": cfg.num_iterations}
    assert run_config.callbacks == []
    assert run_config.checkpoint_config.checkpoint_frequency == 0
    assert run_config.checkpoint_config.checkpoint_score_attribute == cfg.checkpoint_metric


def test_checkpoint_source_metric_defaults_to_tune_metric() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--disable-aim",
                "--metric",
                "custom_metrics/score_mean",
            ]
        )
    )

    assert cfg.metric == "custom_metrics/score_mean"
    assert cfg.checkpoint_source_metric == cfg.metric


def test_callbacks_mirror_nested_checkpoint_metric_to_top_level() -> None:
    cfg = build_train_config(parse_args(["--ray-address", "local", "--disable-aim"]))
    callbacks = build_callbacks_class(cfg)()
    result = {
        "env_runners": {
            "episode_reward_mean": 12.5,
        }
    }

    callbacks.on_train_result(algorithm=None, result=result)

    assert result[cfg.checkpoint_metric] == pytest.approx(12.5)


def test_callbacks_aggregate_reward_breakdown_into_custom_metrics() -> None:
    callbacks = RCSSCallbacks()
    episode = _FakeEpisode()

    callbacks.on_episode_start(
        worker=None,
        base_env=None,
        policies={},
        episode=cast(Any, episode),
        env_index=0,
    )

    episode._infos = {
        1: {
            "scores": {"our": 0, "their": 0},
            "step": 1,
            "reward_breakdown": {
                "goal": 0.0,
                "agent_to_ball_shaping": 0.2,
                "time_decay": -0.01,
            },
        }
    }
    callbacks.on_episode_step(
        worker=None,
        base_env=None,
        policies={},
        episode=cast(Any, episode),
        env_index=0,
    )

    episode._infos = {
        1: {
            "scores": {"our": 1, "their": 0},
            "step": 2,
            "reward_breakdown": {
                "goal": 10.0,
                "agent_to_ball_shaping": 0.1,
                "time_decay": -0.01,
            },
        }
    }
    callbacks.on_episode_step(
        worker=None,
        base_env=None,
        policies={},
        episode=cast(Any, episode),
        env_index=0,
    )
    callbacks.on_episode_end(
        worker=None,
        base_env=None,
        policies={},
        episode=cast(Any, episode),
        env_index=0,
    )

    assert episode.custom_metrics["episode_our_score"] == 1.0
    assert episode.custom_metrics["episode_their_score"] == 0.0
    assert episode.custom_metrics["episode_steps"] == 2.0
    assert episode.custom_metrics["reward_goal_total"] == 10.0
    assert episode.custom_metrics["reward_goal_per_step"] == 5.0
    assert episode.custom_metrics["reward_agent_to_ball_shaping_total"] == pytest.approx(0.3)
    assert episode.custom_metrics["reward_agent_to_ball_shaping_per_step"] == pytest.approx(0.15)
    assert episode.custom_metrics["reward_time_decay_total"] == pytest.approx(-0.02)


