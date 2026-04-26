"""Ray Tune training entry point for curriculum-based RCSS RL experiments."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rcss_env.action_mask import ActionMaskResolver
from rcss_env.config import EnvConfig
from rcss_env.env import RCSSEnv

from train.callbacks import RCSSCallbacks
from train.config import TrainConfig
from train.factory import build_env_config
from train.models.fcnet import register as register_model

logger = logging.getLogger(__name__)

ENV_NAME = "rcss_multi_agent"
DEFAULT_POLICY_ID = "default_policy"


def default_policy_mapping_fn(
    agent_id: Any,
    episode: Any,
    worker: Any = None,
    **kwargs: Any,
) -> str:
    return DEFAULT_POLICY_ID


def build_ppo_config(
    train_cfg: TrainConfig,
    env_config: EnvConfig,
) -> PPOConfig:
    """Build a PPO AlgorithmConfig suitable for Ray Tune."""

    def _env_creator(cfg: dict[str, Any]) -> RCSSEnv:
        return RCSSEnv(config=cfg["env_config"])

    register_env(ENV_NAME, _env_creator)
    register_model()

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=ENV_NAME,
            env_config={"env_config": env_config},
            action_mask_key=ActionMaskResolver.OBSERVATION_KEY,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=train_cfg.num_env_runners,
            num_envs_per_env_runner=train_cfg.num_envs_per_runner,
            num_cpus_per_env_runner=train_cfg.num_cpus_per_runner,
            num_gpus_per_env_runner=0,

        )
        .training(
            train_batch_size=train_cfg.train_batch_size,
            minibatch_size=train_cfg.sgd_minibatch_size,
            num_epochs=train_cfg.num_sgd_iter,
            lr=train_cfg.lr,
            lambda_=0.95,
            vf_clip_param=50.0,
            gamma=train_cfg.gamma,
            entropy_coeff=train_cfg.entropy_coeff,
            clip_param=train_cfg.clip_param,
            model={
                "custom_model": "rcss_fcnet",
                "custom_model_config": {
                    "hidden_sizes": [256, 256],
                },
            },
        )
        .callbacks(RCSSCallbacks)
        .framework("torch")
        .multi_agent(
            policies={DEFAULT_POLICY_ID},
            policy_mapping_fn=default_policy_mapping_fn,
        )
    )

    return config


def build_tune_callbacks(train_cfg: TrainConfig) -> list[Any]:
    callbacks: list[Any] = []
    if train_cfg.enable_aim:
        try:
            from ray.tune.logger.aim import AimLoggerCallback

            callbacks.append(
                AimLoggerCallback(
                    repo=train_cfg.aim_repo,
                    experiment_name=train_cfg.aim_experiment_name or train_cfg.experiment_name,
                    metrics=list(train_cfg.aim_metrics) if train_cfg.aim_metrics else None,
                )
            )
        except (AssertionError, ImportError) as exc:
            raise RuntimeError(
                "Aim logging is enabled, but the 'aim' package is unavailable. "
                "Install dependencies in a Python <3.13 environment or pass --disable-aim for local debugging."
            ) from exc
    return callbacks


def build_run_config(train_cfg: TrainConfig) -> tune.RunConfig:
    checkpoint_config = tune.CheckpointConfig(
        num_to_keep=train_cfg.checkpoint_num_to_keep,
        checkpoint_score_attribute=train_cfg.metric,
        checkpoint_score_order=train_cfg.mode,
        checkpoint_frequency=train_cfg.checkpoint_freq,
        checkpoint_at_end=train_cfg.checkpoint_at_end,
    )

    return tune.RunConfig(
        name=train_cfg.experiment_name,
        storage_path=train_cfg.storage_path,
        stop={"training_iteration": train_cfg.num_iterations},
        checkpoint_config=checkpoint_config,
        callbacks=build_tune_callbacks(train_cfg),
        log_to_file=train_cfg.log_to_file,
    )


def build_tune_config(train_cfg: TrainConfig) -> tune.TuneConfig:
    return tune.TuneConfig(
        metric=train_cfg.metric,
        mode=train_cfg.mode,
        num_samples=train_cfg.num_samples,
    )


def build_tuner(train_cfg: TrainConfig) -> tune.Tuner:
    env_config = build_env_config(train_cfg)
    ppo_config = build_ppo_config(train_cfg, env_config)
    ppo_config.validate()
    param_space = ppo_config.to_dict()

    if train_cfg.restore_path:
        return tune.Tuner.restore(
            train_cfg.restore_path,
            trainable=train_cfg.algo,
            param_space=param_space,
        )

    return tune.Tuner(
        train_cfg.algo,
        param_space=param_space,
        tune_config=build_tune_config(train_cfg),
        run_config=build_run_config(train_cfg),
    )


def init_ray(train_cfg: TrainConfig) -> None:
    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    ray_address = train_cfg.ray_address
    if ray_address and ray_address.lower() not in {"local", "none"}:
        init_kwargs["address"] = ray_address
    ray.init(**init_kwargs)


def _optional_int(value: str) -> int | None:
    if value.lower() in {"", "none", "null"}:
        return None
    return int(value)


def _csv_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TrainConfig(timestamp_experiment_name=False)
    parser = argparse.ArgumentParser(
        description="Train PPO on RCSSEnv through Ray Tune and a selected curriculum."
    )

    # Ray / Tune runtime
    parser.add_argument("--ray-address", type=str, default=defaults.ray_address)
    parser.add_argument("--experiment-name", type=str, default=defaults.experiment_name)
    parser.add_argument("--storage-path", type=str, default=defaults.storage_path)
    parser.add_argument("--restore", dest="restore_path", type=str, default=defaults.restore_path)
    parser.add_argument("--timestamp-experiment-name", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--metric", type=str, default=defaults.metric)
    parser.add_argument("--mode", choices=["min", "max"], default=defaults.mode)
    parser.add_argument("--log-to-file", action="store_true", default=defaults.log_to_file)

    # PPO / RLlib hyperparameters
    parser.add_argument("--num-env-runners", type=int, default=defaults.num_env_runners)
    parser.add_argument("--num-envs-per-runner", type=int, default=defaults.num_envs_per_runner)
    parser.add_argument("--num-cpus-per-runner", type=float, default=defaults.num_cpus_per_runner)
    parser.add_argument("--train-batch-size", type=int, default=defaults.train_batch_size)
    parser.add_argument("--sgd-minibatch-size", type=int, default=defaults.sgd_minibatch_size)
    parser.add_argument("--num-sgd-iter", type=int, default=defaults.num_sgd_iter)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--entropy-coeff", type=float, default=defaults.entropy_coeff)
    parser.add_argument("--clip-param", type=float, default=defaults.clip_param)
    parser.add_argument("--num-iterations", type=int, default=defaults.num_iterations)
    parser.add_argument("--checkpoint-freq", type=int, default=defaults.checkpoint_freq)
    parser.add_argument("--checkpoint-num-to-keep", type=_optional_int, default=defaults.checkpoint_num_to_keep)
    parser.add_argument("--no-checkpoint-at-end", dest="checkpoint_at_end", action="store_false", default=defaults.checkpoint_at_end)

    # Infrastructure
    parser.add_argument("--grpc-host", type=str, default=defaults.grpc_host)
    parser.add_argument("--grpc-port", type=int, default=defaults.grpc_port)
    parser.add_argument("--allocator-host", type=str, default=defaults.allocator_host)
    parser.add_argument("--allocator-port", type=int, default=defaults.allocator_port)

    # Curriculum
    parser.add_argument("--curriculum", choices=["shooting"], default=defaults.curriculum)
    parser.add_argument("--curriculum-debug", action=argparse.BooleanOptionalAction, default=defaults.curriculum_debug)
    parser.add_argument("--agent-unum", type=int, default=defaults.agent_unum)
    parser.add_argument("--team-side", choices=["left", "right", "rand"], default=defaults.team_side)
    parser.add_argument("--our-player-num", type=int, default=defaults.our_player_num)
    parser.add_argument("--oppo-player-num", type=int, default=defaults.oppo_player_num)
    parser.add_argument("--our-goalie-unum", type=_optional_int, default=defaults.our_goalie_unum)
    parser.add_argument("--oppo-goalie-unum", type=_optional_int, default=defaults.oppo_goalie_unum)
    parser.add_argument("--our-team-name", type=str, default=defaults.our_team_name)
    parser.add_argument("--oppo-team-name", type=str, default=defaults.oppo_team_name)
    parser.add_argument("--agent-image", dest="player_agent_image", type=str, default=defaults.player_agent_image)
    parser.add_argument("--bot-image", dest="player_bot_image", type=str, default=defaults.player_bot_image)
    parser.add_argument("--time-up", type=int, default=defaults.time_up)
    parser.add_argument("--goal-l", type=_optional_int, default=defaults.goal_l)
    parser.add_argument("--goal-r", type=_optional_int, default=defaults.goal_r)
    parser.add_argument("--reward-goal", type=float, default=defaults.reward_goal)
    parser.add_argument("--reward-concede", type=float, default=defaults.reward_concede)
    parser.add_argument("--reward-out-of-bounds", type=float, default=defaults.reward_out_of_bounds)
    parser.add_argument("--reward-ball-to-goal-shaping", type=float, default=defaults.reward_ball_to_goal_shaping)
    parser.add_argument("--reward-time-decay", type=float, default=defaults.reward_time_decay)

    # Aim
    parser.add_argument("--disable-aim", dest="enable_aim", action="store_false", default=defaults.enable_aim)
    parser.add_argument("--aim-repo", type=str, default=defaults.aim_repo)
    parser.add_argument("--aim-experiment-name", type=str, default=None)
    parser.add_argument("--aim-metrics", type=str, default=None, help="Comma-separated Tune metric names to log to Aim.")

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args(argv)


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        ray_address=args.ray_address,
        experiment_name=args.experiment_name,
        storage_path=args.storage_path,
        restore_path=args.restore_path,
        timestamp_experiment_name=args.timestamp_experiment_name,
        num_samples=args.num_samples,
        metric=args.metric,
        mode=args.mode,
        log_to_file=args.log_to_file,
        num_env_runners=args.num_env_runners,
        num_envs_per_runner=args.num_envs_per_runner,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.sgd_minibatch_size,
        num_sgd_iter=args.num_sgd_iter,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coeff=args.entropy_coeff,
        clip_param=args.clip_param,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_num_to_keep=args.checkpoint_num_to_keep,
        checkpoint_at_end=args.checkpoint_at_end,
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        allocator_host=args.allocator_host,
        allocator_port=args.allocator_port,
        curriculum=args.curriculum,
        curriculum_debug=args.curriculum_debug,
        agent_unum=args.agent_unum,
        team_side=args.team_side,
        our_player_num=args.our_player_num,
        oppo_player_num=args.oppo_player_num,
        our_goalie_unum=args.our_goalie_unum,
        oppo_goalie_unum=args.oppo_goalie_unum,
        our_team_name=args.our_team_name,
        oppo_team_name=args.oppo_team_name,
        player_agent_image=args.player_agent_image,
        player_bot_image=args.player_bot_image,
        time_up=args.time_up,
        goal_l=args.goal_l,
        goal_r=args.goal_r,
        reward_goal=args.reward_goal,
        reward_concede=args.reward_concede,
        reward_out_of_bounds=args.reward_out_of_bounds,
        reward_ball_to_goal_shaping=args.reward_ball_to_goal_shaping,
        reward_time_decay=args.reward_time_decay,
        enable_aim=args.enable_aim,
        aim_repo=args.aim_repo,
        aim_experiment_name=args.aim_experiment_name,
        aim_metrics=_csv_tuple(args.aim_metrics),
    )


def log_best_result(results: tune.ResultGrid, train_cfg: TrainConfig) -> None:
    if results.errors:
        logger.error("Tune finished with %d errored trial(s)", len(results.errors))
        for error in results.errors:
            logger.error("Tune trial error: %s", error)

    try:
        best = results.get_best_result(metric=train_cfg.metric, mode=train_cfg.mode)
    except RuntimeError as exc:
        logger.warning("Could not determine best Tune result: %s", exc)
        return

    logger.info(
        "Best trial finished: metric=%s mode=%s value=%s checkpoint=%s",
        train_cfg.metric,
        train_cfg.mode,
        best.metrics.get(train_cfg.metric),
        best.checkpoint,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_cfg = build_train_config(args)
    logger.info("Starting Tune training experiment=%s curriculum=%s", train_cfg.experiment_name, train_cfg.curriculum)

    init_ray(train_cfg)
    try:
        tuner = build_tuner(train_cfg)
        results = tuner.fit()
        log_best_result(results, train_cfg)
        return 1 if results.errors else 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
