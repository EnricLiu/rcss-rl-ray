from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.policy.policy import PolicySpec

from rcss_rl.callbacks import RCSSCallbacks
from rcss_rl.config import EnvSchema, PlayerSchema, TrainConfig
from rcss_rl.env.rcss_env import RCSSEnv
from rcss_rl.models.fcnet import register as register_model

logger = logging.getLogger(__name__)

_ALGO_CONFIG_MAP = {
    "PPO": PPOConfig,
    "IMPALA": ImpalaConfig,
}

def build_algo(cfg: TrainConfig) -> ray.rllib.algorithms.Algorithm:

    algo_name = cfg.algo.upper()
    if algo_name not in _ALGO_CONFIG_MAP:
        supported = ", ".join(_ALGO_CONFIG_MAP)
        raise ValueError(
            f"Unsupported algorithm '{cfg.algo}'. Choose from: {supported}"
        )

    env_cfg_dict = dataclasses.asdict(cfg.env_config)

    agent_ids = [f"left_{i}" for i in range(cfg.env_config.num_left)] + [
        f"right_{i}" for i in range(cfg.env_config.num_right)
    ]
    policies = {agent_id: PolicySpec() for agent_id in agent_ids}

    algo_config = (
        _ALGO_CONFIG_MAP[algo_name]()
        .environment(
            env=RCSSEnv,
            env_config=env_cfg_dict,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .env_runners(
            num_env_runners=cfg.num_env_runners,
            num_envs_per_env_runner=cfg.num_envs_per_runner,
        )
        .training(
            gamma=cfg.gamma,
            lr=cfg.lr,
            train_batch_size=cfg.train_batch_size,
        )
        .callbacks(RCSSCallbacks)
        .framework("torch")
    )

    if algo_name == "PPO":
        algo_config = algo_config.training(
            sgd_minibatch_size=cfg.sgd_minibatch_size,
            num_sgd_iter=cfg.num_sgd_iter,
            clip_param=cfg.clip_param,
            entropy_coeff=cfg.entropy_coeff,
        )

    return algo_config.build()

def train(cfg: TrainConfig | None = None) -> None:

    if cfg is None:
        cfg = TrainConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    register_model()
    ray.init(ignore_reinit_error=True)

    algo = build_algo(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for iteration in range(1, cfg.num_iterations + 1):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get(
            "episode_reward_mean", float("nan")
        )
        logger.info(
            "Iteration %d/%d | mean_reward=%.4f",
            iteration,
            cfg.num_iterations,
            mean_reward,
        )

        if cfg.checkpoint_freq > 0 and iteration % cfg.checkpoint_freq == 0:
            path = algo.save(cfg.checkpoint_dir)
            logger.info("Checkpoint saved → %s", path)

    algo.stop()
    ray.shutdown()

def _parse_args(argv: list[str] | None = None) -> TrainConfig:

    parser = argparse.ArgumentParser(
        description="Train RCSS multi-agent RL policy with Ray/RLlib"
    )
    parser.add_argument("--algo", default="PPO", choices=list(_ALGO_CONFIG_MAP))
    parser.add_argument("--iterations", type=int, default=100, dest="num_iterations")
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--num-envs-per-runner", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-left", type=int, default=3)
    parser.add_argument("--num-right", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    env_cfg = EnvSchema(
        ally_players=[
            PlayerSchema(unum=i, goalie=(i == 1), policy_kind="agent")
            for i in range(1, args.num_left + 1)
        ],
        opponent_players=[
            PlayerSchema(unum=i, goalie=(i == 1), policy_kind="bot")
            for i in range(1, args.num_right + 1)
        ],
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    return TrainConfig(
        algo=args.algo,
        num_iterations=args.num_iterations,
        num_env_runners=args.num_env_runners,
        num_envs_per_runner=args.num_envs_per_runner,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        gamma=args.gamma,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        env_config=env_cfg,
    )

def main(argv: list[str] | None = None) -> None:

    train(_parse_args(argv))

if __name__ == "__main__":
    main(sys.argv[1:])
