# RCSS Curriculum PPO Training Template

## Quick Start

```bash
# Default shooting-curriculum PPO training through Ray Tune.
# Aim logging is enabled by default.
python -m train.train

# Typical cluster run: connect to the Ray head and store Tune results on shared storage.
python -m train.train \
    --ray-address auto \
    --storage-path /mnt/ray-results \
    --experiment-name shooting-ppo \
    --aim-repo /mnt/aim \
    --our-player-num 2 \
    --oppo-player-num 2 \
    --num-env-runners 4 \
    --train-batch-size 8000 \
    --sgd-minibatch-size 256 \
    --num-sgd-iter 15 \
    --lr 5e-4 \
    --gamma 0.995 \
    --entropy-coeff 0.005 \
    --clip-param 0.2 \
    --num-iterations 500 \
    --checkpoint-freq 20

# Local config/debug run without Aim. Use a writable local Tune storage path.
python -m train.train \
    --ray-address local \
    --disable-aim \
    --storage-path /tmp/ray-results \
    --num-env-runners 0 \
    --num-iterations 1

# Restore an existing Tune experiment.
python -m train.train --restore /mnt/ray-results/shooting-ppo
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                  train/train.py (Entry Point)            │
│  1. Parse CLI args → TrainConfig                         │
│  2. Build ShootingCurriculum → EnvConfig                 │
│  3. Register RCSSEnv + RCSSFCNet                         │
│  4. Build PPOConfig and launch ray.tune.Tuner            │
│  5. Attach AimLoggerCallback when enabled                │
└──────────────┬───────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐       ┌─────────────────────────┐
    │     Ray Tune        │──────▶│ Aim / AimStack metrics  │
    │  trials/checkpoints │       │ via Tune callback       │
    └──────────┬──────────┘       └─────────────────────────┘
               │
    ┌──────────▼──────────┐
    │     RLlib PPO       │
    │  env_runners × N    │
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
| **TrainConfig**     | `train/config.py`       | Tune, PPO, infrastructure, curriculum, and Aim config |
| **Factory helpers** | `train/factory.py`      | Builds allocator, gRPC, curriculum, and env configs |
| **EnvConfig**       | `rcss_env/config.py`    | Env connection config (gRPC + allocator + curriculum) |
| **RCSSEnv**         | `rcss_env/env.py`       | MultiAgentEnv: reset/step/close lifecycle         |
| **ShootingCurriculum** | `train/curriculum/shooting/` | Builds room schema and reward for shooting tasks |
| **RCSSFCNet**       | `train/models/fcnet.py` | Custom FC network: trunk -> policy head + value head |
| **RCSSCallbacks**   | `train/callbacks.py`    | Episode metrics + mirrored top-level checkpoint score logging |
| **Action**          | `rcss_env/action.py`    | Hybrid discrete+continuous -> protobuf mapping    |
| **Observation**     | `rcss_env/obs.py`       | WorldModel -> 124-d normalised feature vector     |
| **Reward**          | `rcss_env/reward.py`    | Goal-difference reward function                   |

## CLI Arguments Reference

### Ray / Tune

| Argument | Default | Description |
|----------|---------|-------------|
| `--ray-address` | `auto` | Ray address; use `local` or `none` for local `ray.init()` |
| `--experiment-name` | `rcss-shooting` | Tune experiment name prefix |
| `--storage-path` | `/mnt/ray/storage` | Shared Tune results/checkpoint root |
| `--restore` | None | Restore an existing Tune experiment path |
| `--timestamp-experiment-name` / `--no-timestamp-experiment-name` | true | Append a local timestamp to the Tune experiment name |
| `--num-samples` | 1 | Number of Tune samples/trials |
| `--metric` | `env_runners/episode_reward_mean` | Tune optimization / best-trial metric |
| `--checkpoint-metric` | `checkpoint_score` | Top-level mirrored metric used for checkpoint retention |
| `--checkpoint-source-metric` | `env_runners/episode_reward_mean` | Source metric path mirrored into `--checkpoint-metric` |
| `--mode` | `max` | Tune metric mode |
| `--checkpoint-freq` | 10 | Checkpoint every N training iterations |
| `--checkpoint-num-to-keep` | 3 | Number of checkpoints to keep; pass `none` for unlimited |
| `--no-checkpoint-at-end` | false | Disable final checkpoint |

### PPO Hyper-parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-env-runners` | 2 | Parallel env-runner workers |
| `--num-envs-per-runner` | 1 | Vectorised envs per runner |
| `--train-batch-size` | 4000 | Transitions per PPO update |
| `--sgd-minibatch-size` | 128 | PPO minibatch size |
| `--num-sgd-iter` | 10 | PPO epochs per iteration |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--entropy-coeff` | 0.01 | Entropy coefficient |
| `--clip-param` | 0.3 | PPO clipping parameter |
| `--num-iterations` | 100 | Tune stop condition by training iteration |

### Infrastructure

| Argument | Default | Description |
|----------|---------|-------------|
| `--grpc-host` | `0.0.0.0` | gRPC listen address for simulator sidecars |
| `--grpc-port` | 50051 | gRPC listen port |
| `--allocator-host` | `rcss-env-allocator.rcss-gateway-dev.svc.cluster.local` | Allocator service host |
| `--allocator-port` | 80 | Allocator service port |

### Shooting Curriculum

| Argument | Default | Description |
|----------|---------|-------------|
| `--curriculum` | `shooting` | Curriculum registry key |
| `--curriculum-debug` / `--no-curriculum-debug` | true | Enable curriculum debug mode |
| `--agent-unum` | 1 | Controlled player uniform number |
| `--team-side` | `left` | Agent side: `left`, `right`, or `rand` |
| `--our-player-num` | 2 | Number of players on the learning team |
| `--oppo-player-num` | 2 | Number of players on the opponent team |
| `--our-goalie-unum` | 1 | Learning-team goalie unum; pass `none` to disable |
| `--oppo-goalie-unum` | 1 | Opponent goalie unum; pass `none` to disable |
| `--our-team-name` | `nexus-prime` | Learning-team name |
| `--oppo-team-name` | `bot` | Opponent-team name |
| `--agent-image` | `Cyrus2D/SoccerSimulationProxy` | Agent player image |
| `--bot-image` | `HELIOS/helios-base` | Bot player image |
| `--time-up` | 5000 | Episode time limit |
| `--goal-l` | 1 | Stop after left-side goals; pass `none` to disable |
| `--goal-r` | 1 | Stop after right-side goals; pass `none` to disable |
| `--reward-goal` | 10.0 | Sparse reward for each goal scored by the learning team |
| `--reward-concede` | 10.0 | Sparse penalty magnitude for each goal conceded |
| `--reward-out-of-bounds` | 1.0 | Penalty magnitude when the ball exits the field outside the goal mouth |
| `--reward-ball-to-goal-shaping` | 1.0 | Weight for dense ball-to-goal progress shaping |
| `--reward-time-decay` | 0.001 | Per-cycle penalty encouraging faster scoring |

### Aim and Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--disable-aim` | false | Disable Tune Aim callback |
| `--aim-repo` | None | Aim repository path or URI |
| `--aim-experiment-name` | experiment name | Aim experiment name override; defaults to the final Tune experiment name |
| `--aim-metrics` | None | Comma-separated Tune metric allowlist |
| `--log-to-file` | false | Ask Tune to redirect logs to files |
| `--log-level` | `INFO` | Python logging level |

## Notes

- The training path currently supports PPO and uses the legacy RLlib ModelV2 stack because `RCSSFCNet` extends `TorchModelV2`.
- Aim logging is enabled by default. The project declares `aim` for Python versions below 3.13 because Aim's native dependency is not available for CPython 3.13 from PyPI.
- `ShootingReward.compute()` uses full-information `truth` world models where available, combining sparse score deltas, non-goal out-of-bounds penalty, ball-to-goal progress shaping, and a small cycle-based time decay.

