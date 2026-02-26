# rcss-rl-ray

Multi-agent reinforcement learning for **RoboCup Soccer Simulation** (RCSS)
built on top of [Ray](https://ray.io/) and [RLlib](https://docs.ray.io/en/latest/rllib/).

## Overview

```
rcss-rl-ray/
├── rcss_rl/
│   ├── env/
│   │   └── rcss_env.py   # RLlib MultiAgentEnv — the RCSS simulation stub
│   ├── models/
│   │   └── fcnet.py      # Custom shared-trunk FC model (registered as "rcss_fcnet")
│   ├── callbacks.py      # RLlib training callbacks (episode metrics)
│   ├── config.py         # EnvConfig / TrainConfig dataclasses
│   └── train.py          # Training entry point (CLI + importable API)
└── tests/
    ├── test_env.py        # Environment unit tests
    └── test_train.py      # Config / builder unit tests
```

## Requirements

* Python ≥ 3.10
* `ray[rllib]` ≥ 2.10
* `gymnasium` ≥ 1.0
* `torch` ≥ 2.0

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # installs the `rcss-train` CLI script
```

## Quick start

```bash
# Train with defaults (PPO, 100 iterations, 3v3)
rcss-train

# Customise from the command line
rcss-train --algo PPO --iterations 200 --num-env-runners 4 \
           --num-left 5 --num-right 5 --seed 42
```

Or use the Python API directly:

```python
import ray
from rcss_rl.config import EnvConfig, TrainConfig
from rcss_rl.train import train

cfg = TrainConfig(
    algo="PPO",
    num_iterations=50,
    env_config=EnvConfig(num_left=3, num_right=3, seed=0),
)
train(cfg)
```

## Environment

`RCSSEnv` is an RLlib
[`MultiAgentEnv`](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-environments)
with the following spaces (per agent):

| Space             | Type                  | Shape / Size |
|-------------------|-----------------------|--------------|
| Observation space | `Box(float32)`        | `(18,)`      |
| Action space      | `Discrete`            | `6`          |

**Actions:** `dash_forward`, `dash_left`, `dash_right`, `turn_left`, `turn_right`, `kick`.

**Reward:** dense proximity-to-ball reward plus a goal-scored bonus (configurable via `EnvConfig`).

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```
