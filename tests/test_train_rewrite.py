from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
import pytest
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from rcss_env.action import Action
from rcss_env.action_mask import ActionMaskResolver
from train.callbacks import AimCallback, RCSSCallbacks
from train.factory import build_env_config
from train.models.fcnet import RCSSPPOTorchRLModule
from train.train import (
    DEFAULT_POLICY_ID,
    ENV_NAME,
    build_callbacks_class,
    build_rl_module_spec,
    build_tune_run_kwargs,
    build_ppo_config,
    build_run_config,
    build_train_config,
    build_tune_callbacks,
    build_tune_config,
    parse_args,
    run_training,
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
            "--num-learners",
            "2",
            "--num-cpus-per-learner",
            "1.5",
            "--num-gpus-per-learner",
            "0.25",
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
    assert cfg.num_learners == 2
    assert cfg.num_cpus_per_learner == 1.5
    assert cfg.num_gpus_per_learner == 0.25
    assert cfg.our_goalie_unum is None
    assert cfg.goal_r is None


def test_build_train_config_parses_resume_from_checkpoint() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--disable-aim",
                "--resume-from-checkpoint",
                "/tmp/checkpoint_000123",
            ]
        )
    )

    assert cfg.resume_from_checkpoint == "/tmp/checkpoint_000123"


def test_restore_and_resume_from_checkpoint_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_config(
            parse_args(
                [
                    "--ray-address",
                    "local",
                    "--disable-aim",
                    "--restore",
                    "/tmp/exp",
                    "--resume-from-checkpoint",
                    "/tmp/checkpoint_000123",
                ]
            )
        )


def test_resume_from_checkpoint_requires_single_sample() -> None:
    with pytest.raises(ValueError, match="single trial"):
        build_train_config(
            parse_args(
                [
                    "--ray-address",
                    "local",
                    "--disable-aim",
                    "--num-samples",
                    "2",
                    "--resume-from-checkpoint",
                    "/tmp/checkpoint_000123",
                ]
            )
        )


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


def test_build_ppo_config_uses_new_api_stack_and_rlmodule() -> None:
    cfg = build_train_config(parse_args(["--ray-address", "local", "--disable-aim"]))
    env_cfg = build_env_config(cfg)

    ppo_config = build_ppo_config(cfg, env_cfg)

    assert ppo_config.env == ENV_NAME
    assert ppo_config.env_config == {"env_config": env_cfg}
    assert ppo_config.disable_env_checking is True
    assert ppo_config.enable_rl_module_and_learner is True
    assert ppo_config.enable_env_runner_and_connector_v2 is True
    assert ppo_config.train_batch_size_per_learner == cfg.train_batch_size
    assert ppo_config.num_learners == cfg.num_learners
    assert ppo_config.num_cpus_per_learner == cfg.num_cpus_per_learner
    assert ppo_config.num_gpus_per_learner == cfg.num_gpus_per_learner
    assert ppo_config.model.get("custom_model") is None
    assert set(ppo_config.policies) == {DEFAULT_POLICY_ID}
    assert ppo_config.policy_mapping_fn(1, None, None) == DEFAULT_POLICY_ID
    assert isinstance(ppo_config.rl_module_spec, MultiRLModuleSpec)
    assert ppo_config.rl_module_spec.rl_module_specs[DEFAULT_POLICY_ID].module_class is RCSSPPOTorchRLModule
    assert issubclass(ppo_config.callbacks_class, RCSSCallbacks)
    assert ppo_config.callbacks_class.CHECKPOINT_SCORE_ATTRIBUTE == cfg.checkpoint_metric
    assert ppo_config.callbacks_class.CHECKPOINT_SCORE_SOURCE_ATTRIBUTE == cfg.checkpoint_source_metric


def test_rlmodule_applies_discrete_action_mask_only() -> None:
    spec = build_rl_module_spec().rl_module_specs[DEFAULT_POLICY_ID]
    module = spec.build()
    obs_dim = spec.observation_space.shape[0]
    batch = {
        Columns.OBS: {
            "obs": torch.zeros((2, obs_dim), dtype=torch.float32),
            ActionMaskResolver.OBSERVATION_KEY: torch.tensor(
                [
                    [1, 0, 1, 1, 0],
                    [0, 1, 1, 0, 1],
                ],
                dtype=torch.float32,
            ),
        }
    }

    output = module.forward_exploration(batch)
    logits = output[Columns.ACTION_DIST_INPUTS]

    assert logits.shape[-1] > Action.n_actions()
    assert logits[0, 1] < -1.0e8
    assert logits[0, 4] < -1.0e8
    assert logits[1, 0] < -1.0e8
    assert logits[1, 3] < -1.0e8
    assert torch.all(logits[:, Action.n_actions():] > -1.0e8)


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

    assert cfg.metric == "checkpoint_score"
    assert cfg.checkpoint_source_metric == "env_runners/episode_return_mean"
    assert tune_config.num_samples == 2
    assert tune_config.metric == cfg.metric
    assert run_config.name == "unit-train"
    assert run_config.stop == {"training_iteration": cfg.num_iterations}
    assert run_config.callbacks == []
    assert run_config.checkpoint_config.checkpoint_frequency == 0
    assert run_config.checkpoint_config.checkpoint_score_attribute == cfg.checkpoint_metric


def test_build_tune_callbacks_passes_train_config_to_aim() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--no-timestamp-experiment-name",
                "--experiment-name",
                "unit-train",
                "--lr",
                "0.00005",
                "--aim-metrics",
                "checkpoint_score,learners/rcss_policy/total_loss",
            ]
        )
    )

    callbacks = build_tune_callbacks(cfg)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], AimCallback)
    assert callbacks[0]._run_params["experiment_name"] == "unit-train"
    assert callbacks[0]._run_params["lr"] == pytest.approx(0.00005)
    assert callbacks[0]._run_params["aim_metrics"] == (
        "checkpoint_score",
        "learners/rcss_policy/total_loss",
    )


def test_aim_callback_writes_run_params_through_aim_item_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ray.tune.logger.aim import AimLoggerCallback

    fake_run: dict[str, Any] = {}

    monkeypatch.setattr(
        AimLoggerCallback,
        "_create_run",
        lambda self, trial: fake_run,
    )

    callback = AimCallback(
        repo="/tmp/aim",
        run_params={
            "lr": 5e-5,
            "aim_metrics": ("checkpoint_score", "learners/rcss_policy/total_loss"),
            "nested": {"enabled": True, "items": (1, 2)},
        },
    )

    run = callback._create_run(cast(Any, SimpleNamespace()))

    assert run is fake_run
    assert fake_run["lr"] == pytest.approx(5e-5)
    assert fake_run["aim_metrics"] == [
        "checkpoint_score",
        "learners/rcss_policy/total_loss",
    ]
    assert fake_run["nested"] == {"enabled": True, "items": [1, 2]}


def test_build_tune_run_kwargs_matches_run_and_tune_config() -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--disable-aim",
                "--no-timestamp-experiment-name",
                "--experiment-name",
                "unit-train",
            ]
        )
    )

    param_space = {"lr": 1e-4}
    kwargs = build_tune_run_kwargs(cfg, param_space)

    assert kwargs["name"] == "unit-train"
    assert kwargs["storage_path"] == cfg.storage_path
    assert kwargs["metric"] == cfg.metric
    assert kwargs["mode"] == cfg.mode
    assert kwargs["config"] == param_space
    assert kwargs["num_samples"] == cfg.num_samples
    assert kwargs["stop"] == {"training_iteration": cfg.num_iterations}


def test_run_training_uses_tune_run_for_checkpoint_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_train_config(
        parse_args(
            [
                "--ray-address",
                "local",
                "--disable-aim",
                "--resume-from-checkpoint",
                "/tmp/checkpoint_000123",
                "--no-timestamp-experiment-name",
                "--experiment-name",
                "resume-exp",
            ]
        )
    )

    calls: dict[str, Any] = {}
    sentinel = SimpleNamespace(trials=[])

    monkeypatch.setattr("train.train.build_param_space", lambda train_cfg: {"lr": train_cfg.lr})

    def _fake_tune_run(algo: str, **kwargs: Any) -> Any:
        calls["algo"] = algo
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr("train.train.tune.run", _fake_tune_run)

    result = run_training(cfg)

    assert result is sentinel
    assert calls["algo"] == cfg.algo
    assert calls["kwargs"]["restore"] == "/tmp/checkpoint_000123"
    assert calls["kwargs"]["name"] == "resume-exp"
    assert calls["kwargs"]["config"] == {"lr": cfg.lr}


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

    assert cfg.checkpoint_source_metric is not None
    assert result[cfg.checkpoint_metric] == pytest.approx(12.5)
    assert result[cfg.checkpoint_source_metric] == pytest.approx(12.5)


def test_callbacks_emit_checkpoint_score_before_episode_metrics_exist() -> None:
    cfg = build_train_config(parse_args(["--ray-address", "local", "--disable-aim"]))
    callbacks = build_callbacks_class(cfg)()
    result: dict[str, Any] = {"env_runners/num_env_steps_sampled": 1024.0}

    callbacks.on_train_result(algorithm=None, result=result)

    assert cfg.checkpoint_source_metric is not None
    assert result[cfg.checkpoint_metric] == pytest.approx(0.0)
    assert result[cfg.checkpoint_source_metric] == pytest.approx(0.0)


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


