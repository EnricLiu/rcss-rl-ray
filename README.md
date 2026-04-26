# rcss-rl-ray

Multi-agent reinforcement learning for **RoboCup Soccer Simulation** (RCSS)
built on top of [Ray](https://ray.io/) and [RLlib](https://docs.ray.io/en/latest/rllib/).

> **New API stack**: This project has been migrated to the RLlib
> **new API stack** (`RLModule` / `Learner` / `EnvRunner` / `ConnectorV2`).
> The old-stack flags `enable_rl_module_and_learner=False` and
> `enable_env_runner_and_connector_v2=False` have been removed.
> Custom action masking is now implemented via `RCSSPPORLModule`
> (a `DefaultPPOTorchRLModule` subclass in `train/models/fcnet.py`)
> and custom per-episode metrics are logged through `MetricsLogger`
> inside `RCSSCallbacks`.

## Architecture

```mermaid
flowchart TB
    subgraph Training ["Ray Tune Training (train/train.py)"]
        direction TB
        Tune["Tune Tuner\n(single trial / future search)"]
        Curriculum["Curriculum Factory\n(ShootingCurriculum)"]
        PPO["PPOConfig\ncustom model + action mask"]
        Aim["AimLoggerCallback\n(optional AimStack metrics)"]
        Tune --> Curriculum --> PPO
        Tune -. metrics .-> Aim
    end

    subgraph Model ["Neural Network (models/fcnet.py)"]
        Backbone["Shared Backbone\n(2×256 FC layers)"]
        PolicyHead["Policy Head"]
        ValueHead["Value Head"]
        Backbone --> PolicyHead
        Backbone --> ValueHead
    end

    subgraph Env ["RCSSEnv (rcss_env/env.py)"]
        direction TB
        Reset["reset()"]
        Step["step()"]
        Close["close()"]
    end

    subgraph ActionObs ["Action & Observation"]
        Action["Hybrid Action Space\n(catch / dash / kick /\nmove / tackle / turn)"]
        Obs["Observation Extraction\n(124-dim feature vector)"]
        Reward["Reward Computation\n(goal-based ±1)"]
    end

    subgraph Comm ["Communication Layer"]
        gRPC["gRPC Servicer\n(GameServicer)"]
        BatchQ["Batch Queue\n(state batching)"]
        Alloc["Allocator Client\n(REST / HTTP)"]
    end

    subgraph External ["External Services"]
        Proxy["SoccerSimulation\nProxy (sidecar)"]
        AllocSrv["Room Allocator\nService"]
        RCSS["RCSS Simulator"]
    end

    subgraph RayCluster ["Ray Cluster"]
        Training
        Model
    end

    Callbacks["Metrics Callbacks\n(callbacks.py)"]

    RayCluster -- "obs / reward / done" --> Env
    Env -- "transitions" --> RayCluster
    Env --- ActionObs
    Env --- Comm
    Callbacks -. "metric logging" .-> Training

    gRPC <--> |"protobuf\n(StateMessage /\nActionMessage)"| Proxy
    BatchQ --> gRPC
    Alloc -- "room allocate /\ndeallocate" --> AllocSrv
    Proxy <--> RCSS

    classDef external fill:#e8e8e8,stroke:#999
    class Proxy,AllocSrv,RCSS external
```

### Keywords

- **Multi-Agent Reinforcement Learning (MARL)** — simultaneous training of multiple soccer-playing agents
- **RoboCup Soccer Simulation (RCSS)** — 2D simulated soccer competition environment
- **Ray Tune / RLlib** — distributed experiment orchestration and PPO training
- **Curriculum** — room schema and reward construction selected by training config
- **Hybrid Action Space** — discrete action-type selection combined with continuous parameters
- **gRPC** — high-performance RPC for real-time agent–simulator communication
- **Gymnasium MultiAgentEnv** — standard multi-agent environment interface

