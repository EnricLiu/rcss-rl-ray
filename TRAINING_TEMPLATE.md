# RCSS Multi-Agent PPO Training Template

## Quick Start

```bash
# Default training (11 agents vs 11 bots, 100 iterations)
python train.py

# Custom parameters
python train.py \
    --num-agents 11 \
    --num-env-runners 4 \
    --train-batch-size 8000 \
    --sgd-minibatch-size 256 \
    --num-sgd-iter 15 \
    --lr 5e-4 \
    --gamma 0.995 \
    --entropy-coeff 0.005 \
    --clip-param 0.2 \
    --num-iterations 500 \
    --checkpoint-freq 20 \
    --checkpoint-path ./checkpoints

# Resume from checkpoint
python train.py --restore ./checkpoints/best
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     train.py (Entry Point)               │
│  1. Parse CLI args → TrainConfig + EnvConfig             │
│  2. Register custom model (RCSSFCNet)                    │
│  3. Build PPOConfig with multi-agent setup               │
│  4. Run training loop with checkpointing                 │
└──────────────┬───────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │    Ray PPO Engine   │
    │  (env_runners × N)  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐     ┌─────────────────────────┐
    │    RCSSEnv          │────▶│ AllocatorClient (REST)   │
    │  (MultiAgentEnv)    │     │ allocate / release rooms │
    │                     │     └─────────────────────────┘
    │  obs  = 124-d float │     ┌─────────────────────────┐
    │  act  = hybrid      │────▶│ GameServicer (gRPC)     │
    │  rew  = goal-diff   │     │ send actions / recv obs  │
    └─────────────────────┘     └─────────────────────────┘
```

## Key Components

| Component           | File                    | Description                                      |
|---------------------|-------------------------|--------------------------------------------------|
| **TrainConfig**     | `config/train.py`       | Hyper-parameter dataclass (lr, gamma, batch, etc) |
| **EnvConfig**       | `config/env.py`         | Env connection config (room + gRPC + allocator)   |
| **ServerConfig**    | `config/server.py`      | Generic `host:port` config                        |
| **RCSSEnv**         | `rcss_env/env.py`       | MultiAgentEnv: reset/step/close lifecycle         |
| **RCSSFCNet**       | `models/fcnet.py`       | Custom FC network: trunk → policy head + value head |
| **RCSSCallbacks**   | `callbacks.py`          | Episode-end metric logging (scores, steps)        |
| **RoomSchema**      | `schema/room.py`        | Room definition (teams, stopping, referee)        |
| **Action**          | `rcss_env/action.py`    | Hybrid discrete+continuous → protobuf mapping     |
| **Observation**     | `rcss_env/obs.py`       | WorldModel → 124-d normalised feature vector      |
| **Reward**          | `rcss_env/reward.py`    | Goal-difference reward function                   |

## CLI Arguments Reference

### Training Hyper-parameters

| Argument                | Default  | Description                        |
|-------------------------|----------|------------------------------------|
| `--num-env-runners`     | 2        | Parallel env-runner workers        |
| `--num-envs-per-runner` | 1        | Vectorised envs per runner         |
| `--train-batch-size`    | 4000     | Transitions per SGD update         |
| `--sgd-minibatch-size`  | 128      | SGD mini-batch size                |
| `--num-sgd-iter`        | 10       | SGD epochs per iteration           |
| `--lr`                  | 3e-4     | Learning rate                      |
| `--gamma`               | 0.99     | Discount factor                    |
| `--entropy-coeff`       | 0.01     | Entropy coefficient                |
| `--clip-param`          | 0.3      | PPO clipping parameter             |
| `--num-iterations`      | 100      | Total training iterations          |
| `--checkpoint-freq`     | 10       | Checkpoint every N iterations      |
| `--checkpoint-path`     | checkpoints | Checkpoint directory            |

### Infrastructure

| Argument            | Default           | Description                   |
|---------------------|-------------------|-------------------------------|
| `--num-agents`      | 11                | Number of RL agent players    |
| `--grpc-host`       | 0.0.0.0           | gRPC listen address           |
| `--grpc-port`       | 50051             | gRPC listen port              |
| `--allocator-host`  | localhost          | Allocator service host        |
| `--allocator-port`  | 8080              | Allocator service port        |
| `--time-up`         | 6000              | Episode truncation timestep   |
| `--bot-image`       | rcss-bot:latest   | Bot container image           |
| `--agent-image`     | rcss-agent:latest | Agent container image         |

### Misc

| Argument       | Default | Description                           |
|----------------|---------|---------------------------------------|
| `--restore`    | None    | Path to checkpoint to restore from    |
| `--log-level`  | INFO    | Logging level (DEBUG/INFO/WARN/ERROR) |

