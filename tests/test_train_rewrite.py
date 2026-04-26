from __future__ import annotations

from train.factory import build_env_config
from train.train import (
    DEFAULT_POLICY_ID,
    ENV_NAME,
    build_ppo_config,
    build_run_config,
    build_train_config,
    default_policy_mapping_fn,
    build_tune_callbacks,
    build_tune_config,
    parse_args,
)
from train.curriculum.shooting import ShootingCurriculum


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
                "--reward-time-decay",
                "0.02",
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
    assert env_cfg.curriculum.config.reward_time_decay == 0.02
    assert schema.stopping.time_up == 321
    assert len(schema.teams.agent_team.players) == 3
    assert len(opponent_team.players) == 2


def test_build_ppo_config_uses_new_api_stack() -> None:
    cfg = build_train_config(parse_args(["--ray-address", "local", "--disable-aim"]))
    env_cfg = build_env_config(cfg)

    ppo_config = build_ppo_config(cfg, env_cfg)

    assert ppo_config.env == ENV_NAME
    assert ppo_config.env_config == {"env_config": env_cfg}
    assert ppo_config.disable_env_checking is True
    assert ppo_config.enable_rl_module_and_learner is True
    assert ppo_config.enable_env_runner_and_connector_v2 is True
    assert ppo_config.is_multi_agent is True
    assert DEFAULT_POLICY_ID in ppo_config.policies
    assert default_policy_mapping_fn(agent_id=1, episode=None) == DEFAULT_POLICY_ID

    from train.models.fcnet import RCSSPPORLModule
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    spec = ppo_config.rl_module_spec
    assert isinstance(spec, RLModuleSpec)
    assert spec.module_class is RCSSPPORLModule


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
    assert run_config.name == "unit-train"
    assert run_config.stop == {"training_iteration": cfg.num_iterations}
    assert run_config.callbacks == []
    assert run_config.checkpoint_config.checkpoint_frequency == 0
