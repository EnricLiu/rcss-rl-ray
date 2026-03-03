"""Training hyper-parameter configuration."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyper-parameter set for the RLlib training loop.

    Attributes:
        algo: RLlib algorithm name, e.g. ``"PPO"`` or ``"IMPALA"``.
        num_env_runners: Number of parallel env-runner workers.
        num_envs_per_runner: Number of vectorised envs per runner.
        train_batch_size: Total transitions per SGD update.
        sgd_minibatch_size: Mini-batch size per SGD pass (PPO only).
        num_sgd_iter: SGD epochs per training iteration.
        lr: Optimiser learning rate.
        gamma: Discount factor.
        entropy_coeff: Policy entropy regularisation coefficient.
        clip_param: PPO clipping parameter.
        num_iterations: Total number of training iterations.
        checkpoint_freq: Save a checkpoint every N iterations (0 = disabled).
        checkpoint_path: Directory for storing checkpoints.
    """

    algo: str = "PPO"
    num_env_runners: int = 2
    num_envs_per_runner: int = 1
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    clip_param: float = 0.3
    num_iterations: int = 100
    checkpoint_freq: int = 10
    checkpoint_path: Path = Path("checkpoints")
