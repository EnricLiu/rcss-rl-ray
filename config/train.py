from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainConfig:

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
