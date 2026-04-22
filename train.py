"""PPO training entry-point for the RCSS multi-agent environment.

Usage:
    python train.py                         # use defaults
    python train.py --num-env-runners 4     # override specific hyperparameters

The script:
  1. Parses CLI arguments and builds configuration dataclasses.
  2. Registers the custom model with RLlib's ModelCatalog.
  3. Constructs a Ray PPO algorithm configured for the RCSS multi-agent env.
  4. Runs the training loop with periodic checkpointing.
  5. Saves a final checkpoint at the end of training.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from ipaddress import IPv4Address

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from callbacks import RCSSCallbacks
from client.base.allocator.config import AllocatorConfig
from config import EnvConfig, TrainConfig, ServerConfig
from models.fcnet import register as register_model
from rcss_env import RCSSEnv
from schema import (
    GameServerSchema,
    TeamsSchema,
    TeamSchema,
    TeamSide,
    PlayerSchema,
    StoppingEvents,
    RefereeSchema,
    SspAgentPolicy,
    BotPolicy,
    PolicyKind,
    PolicyAgentKind,
)

logger = logging.getLogger(__name__)



def make_default_room_schema(
    num_agents: int = 11,
    grpc_host: IPv4Address = IPv4Address('127.0.0.1'),
    grpc_port: int = 50051,
    bot_image: str = "HELIOS/helios-base",
    agent_image: str = "Cyrus2D/SoccerSimulationProxy",
    time_up: int = 6000,
) -> GameServerSchema:
    """Create a minimal GameServerSchema with *num_agents* SSP-agent players on the
    left team and 11 scripted bots on the right team.
    """

    # --- Left team: RL agents ---
    agent_players = [
        PlayerSchema(
            unum=i,
            policy=SspAgentPolicy(
                kind=PolicyKind.Agent,
                image=agent_image,
                agent=PolicyAgentKind.Ssp,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
            ),
            goalie=(i == 1),
        )
        for i in range(1, num_agents + 1)
    ]

    left_team = TeamSchema(
        name="RLAgents",
        side=TeamSide.LEFT,
        players=agent_players,
    )

    # --- Right team: scripted bots ---
    bot_players = [
        PlayerSchema(
            unum=i,
            policy=BotPolicy(kind=PolicyKind.Bot, image=bot_image),
            goalie=(i == 1),
        )
        for i in range(1, num_agents + 1)
    ]

    right_team = TeamSchema(
        name="Bots",
        side=TeamSide.RIGHT,
        players=bot_players,
    )

    return GameServerSchema(
        teams=TeamsSchema(left=left_team, right=right_team),
        stopping=StoppingEvents(time_up=time_up),
        referee=RefereeSchema(enable=True), log=True,
    )


def make_env_config(
    grpc_host: IPv4Address = IPv4Address('127.0.0.1'),
    grpc_port: int = 50051,
    allocator_host: str = "localhost",
    allocator_port: int = 5555,
    gs_schema: GameServerSchema = None,
) -> EnvConfig:
    """Assemble an :class:`EnvConfig` from connection parameters."""
    if gs_schema is None:
        gs_schema = make_default_room_schema(grpc_host=grpc_host, grpc_port=grpc_port)

    return EnvConfig(
        room=gs_schema,
        grpc=ServerConfig(host=grpc_host, port=grpc_port),
        allocator=AllocatorConfig(base_url=f"http://{allocator_host}:{allocator_port}"),
    )


def build_ppo_config(
    train_cfg: TrainConfig,
    env_config: EnvConfig,
) -> PPOConfig:
    """Return a fully-configured :class:`PPOConfig` ready for ``.build()``."""

    # Env creator registered with Ray so workers can instantiate it
    def _env_creator(cfg: dict):
        return RCSSEnv(config=cfg["env_config"])

    register_env("rcss_multi_agent", _env_creator)

    config = (
        PPOConfig()
        .environment(
            env="rcss_multi_agent",
            env_config={"env_config": env_config},
        )
        .env_runners(
            num_env_runners=train_cfg.num_env_runners,
            num_envs_per_env_runner=train_cfg.num_envs_per_runner,
        )
        .training(
            train_batch_size_per_learner=train_cfg.train_batch_size,
            minibatch_size=train_cfg.sgd_minibatch_size,
            num_epochs=train_cfg.num_sgd_iter,
            lr=train_cfg.lr,
            gamma=train_cfg.gamma,
            entropy_coeff=train_cfg.entropy_coeff,
            clip_param=train_cfg.clip_param,
        )
        .callbacks(RCSSCallbacks)
        .framework("torch")
        .multi_agent(
            # All agents share a single "default_policy"
            policies={"default_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: "default_policy",
        )
        .rl_module(
            model_config={
                "custom_model": "rcss_fcnet",
                "custom_model_config": {
                    "hidden_sizes": [256, 256],
                },
            },
        )
    )

    return config


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on the RCSS environment")

    # Training hyper-parameters
    p.add_argument("--num-env-runners", type=int, default=2, help="Parallel env-runner workers")
    p.add_argument("--num-envs-per-runner", type=int, default=1, help="Vectorised envs per runner")
    p.add_argument("--train-batch-size", type=int, default=4000, help="Transitions per SGD update")
    p.add_argument("--sgd-minibatch-size", type=int, default=128, help="SGD mini-batch size")
    p.add_argument("--num-sgd-iter", type=int, default=10, help="SGD epochs per iteration")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument("--clip-param", type=float, default=0.3, help="PPO clip parameter")
    p.add_argument("--num-iterations", type=int, default=100, help="Total training iterations")
    p.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint every N iterations")
    p.add_argument("--checkpoint-path", type=str, default="checkpoints", help="Checkpoint directory")

    # Infrastructure
    p.add_argument("--num-agents", type=int, default=11, help="Number of RL agent players")
    p.add_argument("--grpc-host", type=str, default="0.0.0.0", help="gRPC listen address")
    p.add_argument("--grpc-port", type=int, default=50051, help="gRPC listen port")
    p.add_argument("--allocator-host", type=str, default="localhost", help="Allocator service host")
    p.add_argument("--allocator-port", type=int, default=8080, help="Allocator service port")
    p.add_argument("--time-up", type=int, default=6000, help="Episode truncation timestep")
    p.add_argument("--bot-image", type=str, default="rcss-bot:latest", help="Bot container image")
    p.add_argument("--agent-image", type=str, default="rcss-agent:latest", help="Agent container image")

    # Misc
    p.add_argument("--restore", type=str, default=None, help="Path to checkpoint to restore from")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- 1. Initialise Ray ----
    ray.init(ignore_reinit_error=True)

    # ---- 2. Register custom model ----
    register_model()

    # ---- 3. Build configs ----
    train_cfg = TrainConfig(
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
        checkpoint_path=Path(args.checkpoint_path),
    )

    room_schema = make_default_room_schema(
        num_agents=args.num_agents,
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        bot_image=args.bot_image,
        agent_image=args.agent_image,
        time_up=args.time_up,
    )

    env_config = make_env_config(grpc_host=args.grpc_host, grpc_port=args.grpc_port, allocator_host=args.allocator_host,
                                 allocator_port=args.allocator_port, gs_schema=room_schema)

    # ---- 4. Build PPO algorithm ----
    ppo_config = build_ppo_config(train_cfg, env_config)
    algo = ppo_config.build()

    # Optionally restore from a previous checkpoint
    if args.restore:
        logger.info("Restoring from checkpoint: %s", args.restore)
        algo.restore(args.restore)

    # ---- 5. Training loop ----
    checkpoint_dir = train_cfg.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")

    for iteration in range(1, train_cfg.num_iterations + 1):
        result = algo.train()

        # Extract key metrics
        env_runners = result.get("env_runners", {})
        episode_reward_mean = env_runners.get("episode_reward_mean", float("nan"))
        episode_len_mean = env_runners.get("episode_len_mean", float("nan"))

        logger.info(
            "Iter %4d / %d | reward_mean=%.3f | ep_len_mean=%.1f",
            iteration,
            train_cfg.num_iterations,
            episode_reward_mean,
            episode_len_mean,
        )

        # Periodic checkpoint
        if train_cfg.checkpoint_freq > 0 and iteration % train_cfg.checkpoint_freq == 0:
            save_path = algo.save(str(checkpoint_dir))
            logger.info("Checkpoint saved → %s", save_path)

        # Track best model
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_path = algo.save(str(checkpoint_dir / "best"))
            logger.info("New best reward %.3f → %s", best_reward, best_path)

    # ---- 6. Final checkpoint ----
    final_path = algo.save(str(checkpoint_dir / "final"))
    logger.info("Training complete. Final checkpoint → %s", final_path)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()


