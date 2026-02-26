"""Training and environment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Parameters that control the simulated RCSS environment.

    Attributes:
        num_left:            Number of agents on the left (attacking) team.
        num_right:           Number of agents on the right (defending) team.
        max_episode_steps:   Maximum number of steps per episode.
        field_half_x:        Half-length of the field along the x-axis.
                             A goal is scored when the ball crosses ±field_half_x.
        move_speed:          Distance an agent moves per step when dashing.
        kick_radius:         Distance within which an agent can kick the ball.
        kick_power:          Initial ball speed after a kick.
        goal_reward:         Reward bonus per goal scored by the agent's team.
        distance_penalty:    Penalty coefficient for distance from the ball.
        seed:                Optional RNG seed for reproducibility.
    """

    num_left: int = 3
    num_right: int = 3
    max_episode_steps: int = 200
    field_half_x: float = 10.0
    move_speed: float = 0.5
    kick_radius: float = 1.0
    kick_power: float = 2.0
    goal_reward: float = 10.0
    distance_penalty: float = 0.01
    seed: int | None = None


@dataclass
class TrainConfig:
    """Top-level training configuration passed to RLlib.

    Attributes:
        algo:                RLlib algorithm name (e.g. ``"PPO"``, ``"IMPALA"``).
        num_env_runners:     Number of parallel environment runner workers.
        num_envs_per_runner: Number of vectorised envs per runner worker.
        train_batch_size:    Total number of transitions per SGD update.
        sgd_minibatch_size:  Mini-batch size for each SGD step (PPO only).
        num_sgd_iter:        Number of SGD epochs per training iteration.
        lr:                  Optimiser learning rate.
        gamma:               Discount factor.
        entropy_coeff:       Entropy bonus coefficient.
        clip_param:          PPO clipping parameter.
        num_iterations:      Total number of training iterations to run.
        checkpoint_freq:     Save a checkpoint every N iterations (0 = never).
        checkpoint_dir:      Directory where checkpoints are written.
        env_config:          Environment configuration forwarded to :class:`RCSSEnv`.
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
    checkpoint_dir: str = "checkpoints"
    env_config: EnvConfig = field(default_factory=EnvConfig)
