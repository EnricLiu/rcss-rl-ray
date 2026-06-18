"""Ray Tune training entry point for curriculum-based RCSS RL experiments."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from rcss_env import obs as observation
from rcss_env.action import Action
from rcss_env.config import EnvConfig
from rcss_env.env import RCSSEnv

from train.callbacks import RCSSCallbacks
from train.config import TrainConfig, build_train_config, parse_args
from train.factory import build_env_config
from train.models.fcnet import RCSSPPOTorchRLModule

logger = logging.getLogger(__name__)

ENV_NAME = "rcss_multi_agent"
DEFAULT_POLICY_ID = "rcss_policy"


def default_policy_mapping_fn(
    agent_id: Any,
    episode: Any,
    worker: Any = None,
    **kwargs: Any,
) -> str:
    return DEFAULT_POLICY_ID


def build_callbacks_class(train_cfg: TrainConfig):
    class ConfiguredRCSSCallbacks(RCSSCallbacks):
        CHECKPOINT_SCORE_ATTRIBUTE = train_cfg.checkpoint_metric
        CHECKPOINT_SCORE_SOURCE_ATTRIBUTE = train_cfg.checkpoint_source_metric

    return ConfiguredRCSSCallbacks


def build_rl_module_spec() -> MultiRLModuleSpec:
    module_spec = RLModuleSpec(
        module_class=RCSSPPOTorchRLModule,
        observation_space=spaces.Box(
            low=np.full((observation.dim(),), -np.inf, dtype=np.float32),
            high=np.full((observation.dim(),), np.inf, dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=Action.space_schema(),
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
            vf_share_layers=True,
        ),
    )
    return MultiRLModuleSpec(
        rl_module_specs={
            DEFAULT_POLICY_ID: module_spec,
        }
    )


def build_ppo_config(
    train_cfg: TrainConfig,
    env_config: EnvConfig,
) -> PPOConfig:
    """Build a PPO AlgorithmConfig suitable for Ray Tune."""

    def _env_creator(cfg: dict[str, Any]) -> RCSSEnv:
        return RCSSEnv(config=cfg["env_config"])

    register_env(ENV_NAME, _env_creator)

    callbacks_class = build_callbacks_class(train_cfg)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env=ENV_NAME,
            env_config={"env_config": env_config},
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=train_cfg.num_env_runners,
            num_envs_per_env_runner=train_cfg.num_envs_per_runner,
            num_cpus_per_env_runner=cast(Any, train_cfg.num_cpus_per_runner),
            num_gpus_per_env_runner=0,
        )
        .learners(
            num_learners=train_cfg.num_learners,
            num_cpus_per_learner=cast(Any, train_cfg.num_cpus_per_learner),
            num_gpus_per_learner=train_cfg.num_gpus_per_learner,
        )
        .training(
            train_batch_size_per_learner=train_cfg.train_batch_size,
            minibatch_size=train_cfg.sgd_minibatch_size,
            num_epochs=train_cfg.num_sgd_iter,
            lr=train_cfg.lr,
            lambda_=0.95,
            vf_clip_param=50.0,
            gamma=train_cfg.gamma,
            entropy_coeff=train_cfg.entropy_coeff,
            clip_param=train_cfg.clip_param,
        )
        .rl_module(rl_module_spec=build_rl_module_spec())
        .callbacks(callbacks_class)
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
            from train.callbacks import AimCallback

            callbacks.append(
                AimCallback(
                    repo=train_cfg.aim_repo,
                    experiment_name=train_cfg.aim_experiment_name or train_cfg.experiment_name,
                    metrics=list(train_cfg.aim_metrics) if train_cfg.aim_metrics else None,
                    run_params=train_cfg.to_legacy_dict(),
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
        checkpoint_score_attribute=train_cfg.checkpoint_metric,
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


def build_param_space(train_cfg: TrainConfig) -> dict[str, Any]:
    env_config = build_env_config(train_cfg)
    ppo_config = build_ppo_config(train_cfg, env_config)
    ppo_config.validate()
    return ppo_config.to_dict()


def build_tune_run_kwargs(train_cfg: TrainConfig, param_space: dict[str, Any]) -> dict[str, Any]:
    run_config = build_run_config(train_cfg)
    tune_config = build_tune_config(train_cfg)
    return {
        "name": run_config.name,
        "storage_path": run_config.storage_path,
        "metric": tune_config.metric,
        "mode": tune_config.mode,
        "stop": run_config.stop,
        "config": param_space,
        "num_samples": tune_config.num_samples,
        "checkpoint_config": run_config.checkpoint_config,
        "callbacks": run_config.callbacks,
        "log_to_file": run_config.log_to_file,
    }


def build_tuner(train_cfg: TrainConfig) -> tune.Tuner:
    param_space = build_param_space(train_cfg)

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


def run_training(train_cfg: TrainConfig) -> Any:
    if train_cfg.resume_from_checkpoint:
        param_space = build_param_space(train_cfg)
        tune_run_kwargs = build_tune_run_kwargs(train_cfg, param_space)
        logger.info(
            "Starting new Tune experiment from checkpoint=%s",
            train_cfg.resume_from_checkpoint,
        )
        return tune.run(
            train_cfg.algo,
            restore=train_cfg.resume_from_checkpoint,
            **tune_run_kwargs,
        )

    tuner = build_tuner(train_cfg)
    return tuner.fit()


def init_ray(train_cfg: TrainConfig) -> None:
    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    ray_address = train_cfg.ray_address
    if ray_address and ray_address.lower() not in {"local", "none"}:
        init_kwargs["address"] = ray_address
    ray.init(**init_kwargs)


def log_best_result(results: Any, train_cfg: TrainConfig) -> None:
    if hasattr(results, "errors"):
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
        return

    trials = getattr(results, "trials", [])
    errored_trials = [trial for trial in trials if getattr(trial, "status", None) == "ERROR"]
    if errored_trials:
        logger.error("Tune finished with %d errored trial(s)", len(errored_trials))
        for trial in errored_trials:
            logger.error("Tune trial error: %s", getattr(trial, "error_file", trial))

    best_trial = results.get_best_trial(metric=train_cfg.metric, mode=train_cfg.mode)
    if best_trial is None:
        logger.warning("Could not determine best Tune result: no completed trial found")
        return

    metric_value = RCSSCallbacks._lookup_metric(best_trial.last_result, train_cfg.metric)
    checkpoint = results.get_best_checkpoint(best_trial, metric=train_cfg.metric, mode=train_cfg.mode)
    if checkpoint is None:
        checkpoint = results.get_last_checkpoint(best_trial)

    logger.info(
        "Best trial finished: metric=%s mode=%s value=%s checkpoint=%s",
        train_cfg.metric,
        train_cfg.mode,
        metric_value,
        checkpoint,
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
        results = run_training(train_cfg)
        log_best_result(results, train_cfg)
        if hasattr(results, "errors"):
            return 1 if results.errors else 0

        has_errors = any(getattr(trial, "status", None) == "ERROR" for trial in getattr(results, "trials", []))
        return 1 if has_errors else 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
