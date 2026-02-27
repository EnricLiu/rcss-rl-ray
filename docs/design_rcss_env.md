# RCSSEnv 设计方案

## 1. 概述

`RCSSEnv` 是基于 Gymnasium API 的多智能体强化学习环境，用于与 RoboCup Soccer Simulation (rcss) 模拟器交互。它通过以下三个组件协同工作：

1. **AllocatorClient** — 通过 REST API 向 `rcss_cluster` 的 allocator 请求/释放模拟房间
2. **GameServicer (gRPC)** — 通过 gRPC 与 SoccerSimulationProxy sidecar 交换环境状态和动作
3. **RCSSEnv** — 将上述两者整合为标准 Gymnasium 环境，供 Ray/RLlib 训练使用

### 当前问题

目前三个组件之间**完全没有联动**：
- `RCSSEnv` 使用自建的简化物理模拟，不与任何外部服务通信
- `AllocatorClient` 只是独立的 HTTP 客户端，未被 `RCSSEnv` 调用
- `GameServicer` 虽然实现了 gRPC 协议，但未被 `RCSSEnv` 启动或使用
- 模拟房间的配置（template.json）未被传递给 allocator

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Training Process (本项目)                                    │
│                                                             │
│  ┌───────────┐     ┌──────────────────────────────────────┐ │
│  │ Ray/RLlib │────▶│           RCSSEnv                    │ │
│  │ Algorithm │◀────│  (Gymnasium multi-agent env)         │ │
│  └───────────┘     │                                      │ │
│                    │  ┌──────────────┐ ┌───────────────┐  │ │
│                    │  │ Allocator    │ │ GameServicer  │  │ │
│                    │  │ Client       │ │ (gRPC server) │  │ │
│                    │  └──────┬───────┘ └───────┬───────┘  │ │
│                    └─────────┼─────────────────┼──────────┘ │
│                              │ REST            │ gRPC       │
└──────────────────────────────┼─────────────────┼────────────┘
                               │                 │
              ┌────────────────▼──────┐  ┌───────▼───────────────┐
              │   rcss_cluster        │  │ SoccerSimulationProxy │
              │   Allocator (K8s)     │  │ (Sidecar × N)         │
              │                       │  │                       │
              │  管理模拟房间的          │  │  每个 timestep:        │
              │  创建/分配/销毁         │  │  State ──▶ gRPC       │
              └───────────┬───────────┘  │  gRPC ──▶ Actions     │
                          │              └───────────┬───────────┘
                          │ 管理                      │ 代理
                          ▼                           ▼
              ┌───────────────────────────────────────────────┐
              │          rcssserver (模拟器)                    │
              │          足球模拟环境                           │
              └───────────────────────────────────────────────┘
```

## 3. 核心交互流程

### 3.1 单 gRPC 服务器多智能体架构

**关键理解**：每个训练 agent（球员）对应一个独立的 SoccerSimulationProxy sidecar 实例，但所有 sidecar 可以连接到**同一个 gRPC 服务器端口**。原因是 `GetPlayerActions` 接收的 `request: pb2.State` 中包含了球员的身份信息——通过 `request.world_model.self` 可以获取当前请求对应的球员，例如 `request.world_model.self.id` 即为球员的 uniform number。

因此：

- **每个环境实例只需启动一个 gRPC 服务器**，该服务器内的 `GameServicer` 根据请求中的球员身份（unum）进行分发
- 所有训练 agent 的 sidecar 共享同一个 gRPC 地址和端口
- `GameServicer` 内部维护**按 unum 索引**的状态缓冲区和动作缓冲区

```
RCSSEnv
  └── GameServicer (port=50051)
        ├── state_buffers[unum=2]  ◀──  Sidecar[0] (player unum=2)
        ├── state_buffers[unum=3]  ◀──  Sidecar[1] (player unum=3)
        └── state_buffers[unum=7]  ◀──  Sidecar[2] (player unum=7)
        (所有 sidecar 连同一端口，靠 world_model.self.id 区分球员)
```

### 3.2 完整生命周期

```
reset()                                         step()
  │                                               │
  ▼                                               ▼
┌──────────────────────┐                  ┌──────────────────────┐
│ 1. 启动一个 gRPC     │                  │ 1. 等待 servicer 收到 │
│    服务器             │                  │    所有训练agent的State│
│    (GameServicer)    │                  │    (per-unum events)  │
│                      │                  │                       │
│ 2. 构建 RoomRequest  │                  │ 2. 从各 WorldModel    │
│    (含统一gRPC地址)   │                  │    提取观测            │
│                      │                  │    (observation)       │
│ 3. AllocatorClient   │                  │                       │
│    .request_room()   │                  │ 3. 返回 obs 给 RLlib  │
│    → 获得 room_id    │                  │    RL产出 action_dict  │
│                      │                  │                       │
│ 4. 等待所有agent的    │                  │ 4. 将 action 按 unum  │
│    sidecar发来首State │                  │    转换为 PlayerActions│
│                      │                  │    写入 servicer       │
│ 5. 提取初始观测       │                  │    (set_actions)       │
│    返回 obs, info    │                  │                       │
└──────────────────────┘                  │ 5. Sidecar 接收并     │
                                          │    执行动作            │
close()                                   │                       │
  │                                       │ 6. 检测终止/截断条件   │
  ▼                                       │    返回 obs, rew,     │
┌──────────────────────┐                  │    term, trunc, info  │
│ 1. AllocatorClient   │                  └──────────────────────┘
│    .release_room()   │
│                      │
│ 2. 停止 gRPC 服务器  │
│                      │
│ 3. 清理状态          │
└──────────────────────┘
```

## 4. 详细设计

### 4.1 配置扩展 (`EnvConfig`)

```python
@dataclass
class EnvConfig:
    # --- 现有配置 ---
    num_left: int = 3           # 左方队伍总人数 (bot + agent)
    num_right: int = 3          # 右方队伍总人数 (bot + agent)
    max_episode_steps: int = 6000  # 训练端的 episode 截断步数 (可与 time_up 不同)
    # ... 奖励参数等 ...

    # --- 新增：远程模式配置 ---
    mode: str = "local"                       # "local" | "remote"
    allocator_url: str = ""                   # allocator REST endpoint
    grpc_host: str = "0.0.0.0"               # gRPC 服务监听地址
    grpc_port: int = 50051                   # gRPC 服务端口 (每个 env 实例一个)
    grpc_advertise_host: str = "localhost"   # 告知 sidecar 的回连地址

    # --- 新增：队伍构成 ---
    left_agents: list[int] = field(default_factory=lambda: [2, 3, 4])
    # 左方参与训练的球员 uniform numbers
    right_agents: list[int] = field(default_factory=list)
    # 右方参与训练的球员 uniform numbers (通常为空)
    left_bot_image: str = "HELIOS/helios-base"    # 左方 bot 的 Docker 镜像
    right_bot_image: str = "HELIOS/helios-base"   # 右方 bot 的 Docker 镜像
    agent_image: str = "Cyrus2D/SoccerSimulationProxy"  # agent sidecar 的镜像
    agent_type: str = "ssp"                       # agent 类型标识 (如 "ssp")

    # --- 新增：停止条件 (对应 template.json → stopping) ---
    time_up: int = 6000                      # rcssserver 模拟 cycle 上限
    goal_limit_l: int = 0                    # 左方进球达到此数即停 (0=不限)

    # --- 新增：裁判 (对应 template.json → referee) ---
    referee_enable: bool = False             # 是否启用裁判模块

    # --- 新增：初始状态 (对应 template.json → init_state) ---
    ball_init_x: float | None = None         # 球的初始归一化 X 坐标 (0~1)
    ball_init_y: float | None = None         # 球的初始归一化 Y 坐标 (0~1)
```

> **与 template.json 的映射说明**
>
> allocator 期望的 `template.json` 结构来源于 [rcss_cluster/match_composer](https://github.com/EnricLiu/rcss_cluster/blob/sidecar/match_composer/sidecars/match_composer/docs/template.json)。
> `EnvConfig` 中的字段与 template.json 的对应关系：
>
> | EnvConfig 字段 | template.json 路径 | 说明 |
> |---|---|---|
> | `time_up` | `stopping.time_up` | 模拟 cycle 上限 |
> | `goal_limit_l` | `stopping.goal_l` | 左方进球数停止条件 |
> | `referee_enable` | `referee.enable` | 是否启用裁判 |
> | `ball_init_x/y` | `init_state.ball.x/y` | 球初始位置 |
> | `grpc_advertise_host` | 各 agent player → `policy.grpc_host` | sidecar 回连地址 |
> | `grpc_port` | 各 agent player → `policy.grpc_port` | sidecar 回连端口 |
> | `agent_image` | 各 agent player → `policy.image` | SSP 镜像名 |
> | `agent_type` | 各 agent player → `policy.agent` | agent 类型 (如 "ssp") |
> | `left_bot_image` | 左方 bot player → `policy.image` | bot 镜像名 |
> | `right_bot_image` | 右方 bot player → `policy.image` | bot 镜像名 |
>
> per-player 的 `init_state`（位置/体力）和 `blocklist`（禁用动作）在高级场景中由 `PlayerConfig` 单独控制，详见 [§6](#6-playerconfig-与-roomrequest-改进)。

### 4.2 RCSSEnv 改造

```python
class RCSSEnv(gymnasium.Env):
    """支持 local 和 remote 两种模式的多智能体环境"""

    def __init__(self, config):
        # ... 解析配置 ...

        if self._cfg.mode == "remote":
            self._setup_remote()
        # else: 保持现有 local 模式不变

    def _setup_remote(self):
        """初始化远程模式所需的组件"""
        # 单个 GameServicer 处理所有训练 agent
        self._servicer = GameServicer()
        self._grpc_server: grpc.Server | None = None
        self._allocator = AllocatorClient(self._cfg.allocator_url)
        self._room_id: str | None = None

    def reset(self, *, seed=None, options=None):
        if self._cfg.mode == "remote":
            return self._reset_remote(seed=seed, options=options)
        return self._reset_local(seed=seed, options=options)

    def _reset_remote(self, *, seed, options):
        # 1. 如果有旧房间，先释放
        self._cleanup_room()

        # 2. 启动单个 gRPC 服务器
        self._start_grpc_server()

        # 3. 构建 RoomRequest 并请求房间
        room_request = self._build_room_request()
        response = self._allocator.request_room(room_request)
        self._room_id = response["room_id"]

        # 4. 等待所有训练 agent 的 sidecar 发来首个 State
        self._wait_for_initial_states()

        # 5. 从 servicer 中按 unum 提取各 agent 的初始观测
        return self._collect_observations(), self._collect_infos()

    def step(self, action_dict):
        if self._cfg.mode == "remote":
            return self._step_remote(action_dict)
        return self._step_local(action_dict)

    def _step_remote(self, action_dict):
        # 1. 将 action_dict 按 unum 转换为 PlayerActions，写入 servicer
        for agent_id, action in action_dict.items():
            unum = self._agent_id_to_unum(agent_id)
            player_actions = self._action_to_proto(action)
            self._servicer.set_actions(unum, player_actions)

        # 2. 等待所有训练 agent 的 sidecar 发来下一个 State
        self._wait_for_states()

        # 3. 提取观测、奖励、终止条件
        obs = self._collect_observations()
        rewards = self._compute_rewards()
        terminateds, truncateds = self._check_done()
        infos = self._collect_infos()

        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        if self._cfg.mode == "remote":
            self._cleanup_room()
            self._stop_grpc_server()
```

### 4.3 gRPC 服务器管理

每个 `RCSSEnv` 实例只需管理**一个 gRPC 服务器**。`GameServicer` 内部根据 `request.world_model.self.id`（即 unum）区分不同球员的请求。

```python
def _start_grpc_server(self):
    """启动单个 gRPC 服务器，服务所有训练 agent"""
    self._servicer = GameServicer()
    self._grpc_server = serve(
        self._servicer,
        port=self._cfg.grpc_port,
        block=False,
    )

def _stop_grpc_server(self):
    """停止 gRPC 服务器"""
    if self._grpc_server is not None:
        self._grpc_server.stop(grace=5)
        self._grpc_server = None
```

### 4.4 RoomRequest 构建

`RoomRequest` 需要根据环境配置生成符合 `template.json` 格式的请求。
在 template.json 中，每个 `kind: "agent"` 球员的 `policy` 块**单独**携带 `grpc_host`/`grpc_port`、`agent` 类型和 `image` 字段。虽然所有训练 agent 指向同一个 gRPC 服务器，但这些字段需要写在每个 agent 球员的 policy 内部：

```python
def _build_room_request(self) -> RoomRequest:
    """根据配置构建房间请求"""
    left_players = []
    right_players = []

    # 左方队伍
    for unum in range(1, self._cfg.num_left + 1):
        if unum in self._cfg.left_agents:
            # 训练 agent — 每个 agent 的 policy 中包含 gRPC 地址
            left_players.append(PlayerConfig(
                unum=unum,
                goalie=(unum == 1),
                policy_kind="agent",
                policy_agent=self._cfg.agent_type,
                policy_image=self._cfg.agent_image,
                grpc_host=self._cfg.grpc_advertise_host,
                grpc_port=self._cfg.grpc_port,
            ))
        else:
            # Bot 球员
            left_players.append(PlayerConfig(
                unum=unum,
                goalie=(unum == 1),
                policy_kind="bot",
                policy_image=self._cfg.left_bot_image,
            ))

    # 右方队伍同理...

    return RoomRequest(
        ally_name="RLAgent",
        opponent_name="Bot",
        ally_players=left_players,
        opponent_players=right_players,
        time_up=self._cfg.time_up,
        goal_limit_l=self._cfg.goal_limit_l,
        referee_enable=self._cfg.referee_enable,
        ball_init_x=self._cfg.ball_init_x,
        ball_init_y=self._cfg.ball_init_y,
    )
```

### 4.5 template.json 配置映射

allocator 期望的 `template.json` 结构参考自 [rcss_cluster/match_composer](https://github.com/EnricLiu/rcss_cluster/blob/sidecar/match_composer/sidecars/match_composer/docs/template.json)。gRPC 连接信息（`grpc_host`/`grpc_port`）位于**每个 agent 球员的 `policy` 块内**，虽然所有训练 agent 实际指向同一个 gRPC 服务器。此外还包括 `referee`、`init_state`（球的初始位置）、per-player `init_state`（球员初始位置/体力）、`blocklist`（禁用特定动作）等字段：

```json
{
  "api_version": 1,
  "referee": {
    "enable": false
  },
  "stopping": {
    "time_up": 6000,
    "goal_l": 0
  },
  "init_state": {
    "ball": {
      "x": 0.5,
      "y": 0.5
    }
  },
  "teams": {
    "allies": {
      "name": "RLAgent",
      "players": [
        {
          "unum": 1,
          "goalie": true,
          "policy": { "kind": "bot", "image": "HELIOS/helios-base" },
          "init_state": { "pos": { "x": 0.9, "y": 0.5 }, "stamina": 6000 },
          "blocklist": { "dash": true, "catch": false }
        },
        {
          "unum": 2,
          "goalie": false,
          "policy": {
            "kind": "agent",
            "agent": "ssp",
            "image": "Cyrus2D/SoccerSimulationProxy",
            "grpc_host": "127.0.0.1",
            "grpc_port": 6657
          },
          "init_state": { "pos": { "x": 0.7, "y": 0.5 } }
        },
        {
          "unum": 3,
          "goalie": false,
          "policy": {
            "kind": "agent",
            "agent": "ssp",
            "image": "Cyrus2D/SoccerSimulationProxy",
            "grpc_host": "127.0.0.1",
            "grpc_port": 6657
          },
          "init_state": { "pos": { "x": 0.5, "y": 0.5 } }
        }
      ]
    },
    "opponents": {
      "name": "Opponent",
      "players": [
        {
          "unum": 1,
          "goalie": true,
          "policy": { "kind": "bot", "image": "HELIOS/helios-base" },
          "init_state": { "pos": { "x": 0.9, "y": 0.5 }, "stamina": 6000 },
          "blocklist": { "dash": true, "catch": false }
        },
        {
          "unum": 2,
          "goalie": false,
          "policy": { "kind": "bot", "image": "HELIOS/helios-base" },
          "init_state": { "pos": { "x": 0.7, "y": 0.5 } }
        }
      ]
    }
  }
}
```

**关键点**：

- `grpc_host` 和 `grpc_port` 位于**每个 agent 球员的 `policy` 内**（而非顶层），这是 template.json 的实际格式
- 所有训练 agent 的 `grpc_host`/`grpc_port` 可以相同——它们都指向 `RCSSEnv` 启动的同一个 gRPC 服务器
- `GameServicer` 在收到 `GetPlayerActions` 请求时，通过 `request.world_model.self.id`（unum）识别是哪位球员的请求
- 每个 agent player 的 `policy.agent` 字段指定代理类型（如 `"ssp"` 表示 SoccerSimulationProxy）
- 每个 agent player 的 `policy.image` 字段指定 sidecar 的 Docker 镜像
- `init_state` 和 `blocklist` 是可选的 per-player 覆盖配置

### 4.6 Agent ID 映射

需要建立 agent_id ↔ uniform_number 的双向映射：

```python
# agent_id 格式: "{side}_{unum}"
# 例: "left_2" = 左方 2 号球员

def _unum_to_agent_id(self, side: str, unum: int) -> str:
    return f"{side}_{unum}"

def _agent_id_to_unum(self, agent_id: str) -> tuple[str, int]:
    side, unum_str = agent_id.split("_")
    return side, int(unum_str)
```

**注意**：只有配置为 `kind: "agent"` 的球员才是 RL 训练的 agent，bot 球员不参与训练。

### 4.7 观测空间 (Observation)

从 `WorldModel` protobuf 消息提取观测。核心字段：

| 类别 | 来源 | 字段 |
|------|------|------|
| **球信息** | `wm.ball` | `position.x/y`, `velocity.x/y`, `dist_from_self`, `angle_from_self` |
| **自身信息** | `wm.self` | `position.x/y`, `velocity.x/y`, `body_direction`, `stamina`, `is_kickable`, `kick_rate` |
| **队友** | `wm.teammates[i]` | `position.x/y`, `velocity.x/y`, `body_direction`, `uniform_number` |
| **对手** | `wm.opponents[i]` | `position.x/y`, `velocity.x/y`, `body_direction`, `uniform_number` |
| **比赛状态** | `wm` | `cycle`, `game_mode_type`, `our_team_score`, `their_team_score`, `is_our_set_play` |
| **截断信息** | `wm.intercept_table` | `self_reach_steps`, `first_teammate_reach_steps`, `first_opponent_reach_steps` |

```python
def _world_model_to_obs(self, wm: pb2.WorldModel) -> np.ndarray:
    """将 protobuf WorldModel 转换为 numpy 观测向量"""
    obs = []

    # 球
    obs.extend([wm.ball.position.x, wm.ball.position.y,
                wm.ball.velocity.x, wm.ball.velocity.y])

    # 自身
    obs.extend([wm.self.position.x, wm.self.position.y,
                wm.self.velocity.x, wm.self.velocity.y,
                wm.self.body_direction, wm.self.stamina,
                float(wm.self.is_kickable)])

    # 队友 (按 uniform_number 排序，固定长度)
    for tm in sorted(wm.teammates, key=lambda p: p.uniform_number):
        obs.extend([tm.position.x, tm.position.y,
                    tm.velocity.x, tm.velocity.y])

    # 对手 (同理)
    for opp in sorted(wm.opponents, key=lambda p: p.uniform_number):
        obs.extend([opp.position.x, opp.position.y,
                    opp.velocity.x, opp.velocity.y])

    # 比赛状态
    obs.extend([float(wm.cycle), float(wm.game_mode_type),
                float(wm.our_team_score), float(wm.their_team_score)])

    return np.array(obs, dtype=np.float32)
```

### 4.8 动作空间 (Action)

将 RL 输出转换为 protobuf `PlayerActions`。推荐两种设计方案：

#### 方案 A：低级动作（Dash/Turn/Kick）

直接使用 rcss 原生的低级动作，动作空间为连续或混合空间：

```python
# 连续动作空间: [action_type, param1, param2]
# action_type: 0=Dash, 1=Turn, 2=Kick, 3=Tackle, 4=Move
action_space = spaces.Box(low=-1, high=1, shape=(3,))

def _action_to_proto(self, action: np.ndarray) -> pb2.PlayerActions:
    actions = pb2.PlayerActions()
    action_type = int(np.clip(np.round((action[0] + 1) / 2 * 4), 0, 4))
    # Maps [-1,1] → [0,4] uniformly covering all 5 action types

    if action_type == 0:  # Dash
        actions.actions.append(pb2.PlayerAction(
            dash=pb2.Dash(power=action[1]*100, relative_direction=action[2]*180)
        ))
    elif action_type == 1:  # Turn
        actions.actions.append(pb2.PlayerAction(
            turn=pb2.Turn(relative_direction=action[1]*180)
        ))
    elif action_type == 2:  # Kick
        actions.actions.append(pb2.PlayerAction(
            kick=pb2.Kick(power=action[1]*100, relative_direction=action[2]*180)
        ))
    # ...
    return actions
```

#### 方案 B：高级动作（Helios 行为）

使用 SoccerSimulationProxy 提供的高级行为封装：

```python
# 离散动作空间
action_space = spaces.Discrete(N_HIGH_LEVEL_ACTIONS)

ACTIONS = [
    lambda: pb2.PlayerAction(helios_chain_action=pb2.HeliosChainAction(
        direct_pass=True, lead_pass=True, through_pass=True,
        short_dribble=True, long_dribble=True, cross=True,
        simple_pass=True, simple_dribble=True, simple_shoot=True)),
    lambda: pb2.PlayerAction(body_go_to_point=pb2.Body_GoToPoint(...)),
    lambda: pb2.PlayerAction(helios_basic_move=pb2.HeliosBasicMove()),
    # ...
]
```

#### 推荐方案

建议先实现**方案 A（低级动作）**，保持动作空间简单直接。同时附加 `Neck_TurnToBallOrScan` 作为默认的颈部动作。

### 4.9 奖励函数

```python
def _compute_reward(self, agent_id: str,
                    prev_wm: pb2.WorldModel,
                    curr_wm: pb2.WorldModel) -> float:
    reward = 0.0

    # 1. 进球奖励
    score_diff = curr_wm.our_team_score - prev_wm.our_team_score
    opp_score_diff = curr_wm.their_team_score - prev_wm.their_team_score
    reward += score_diff * self._cfg.goal_reward
    reward -= opp_score_diff * self._cfg.goal_reward

    # 2. 接近球的奖励 (可选)
    dist_to_ball = curr_wm.self.dist_from_ball
    reward -= self._cfg.distance_penalty * dist_to_ball

    # 3. 控球奖励 (可选)
    if curr_wm.self.is_kickable:
        reward += self._cfg.possession_reward

    return reward
```

### 4.10 终止与截断

```python
def _check_done(self) -> tuple[dict[str, bool], dict[str, bool]]:
    terminateds = {}
    truncateds = {}

    # 从 servicer 中任一已注册球员获取世界模型
    first_unum = next(iter(self._training_unums))
    state = self._servicer.get_state(first_unum)
    wm = state.world_model if state else None

    # 比赛结束 (game_mode_type == TimeOver)
    game_over = (wm is not None and wm.game_mode_type == pb2.TimeOver)

    # 或达到最大步数
    truncated = (wm is not None and wm.cycle >= self._cfg.max_episode_steps)

    for agent_id in self._training_agent_ids:
        terminateds[agent_id] = game_over
        truncateds[agent_id] = truncated

    terminateds["__all__"] = game_over
    truncateds["__all__"] = truncated

    return terminateds, truncateds
```

## 5. 同步机制

### 5.1 GameServicer 多球员分发

`GameServicer` 内部维护按 unum 索引的缓冲区。当不同球员的 sidecar 并发调用 `GetPlayerActions` 时，servicer 根据 `request.world_model.self.id` 分发到对应的缓冲区槽位：

```python
class GameServicer(service_pb2_grpc.GameServicer):
    def __init__(self) -> None:
        # 按 unum 索引的 per-player 状态和事件
        self._states: dict[int, pb2.State] = {}         # unum → latest State
        self._actions: dict[int, pb2.PlayerActions] = {} # unum → pending Actions
        self._state_events: dict[int, threading.Event] = {}   # unum → state_ready
        self._action_events: dict[int, threading.Event] = {}  # unum → action_ready
        self._lock = threading.Lock()

    def register_player(self, unum: int) -> None:
        """注册一个训练 agent（在 reset 时调用）"""
        self._state_events[unum] = threading.Event()
        self._action_events[unum] = threading.Event()

    def GetPlayerActions(self, request: pb2.State, context) -> pb2.PlayerActions:
        wm = request.world_model
        if wm is None or not wm.HasField("self"):
            logger.warning("Malformed State: missing world_model.self")
            return pb2.PlayerActions()
        unum = wm.self.id
        # 存储该球员的状态
        with self._lock:
            self._states[unum] = request
        # 通知 RCSSEnv: 此球员的状态已到达
        self._state_events[unum].set()

        # 等待 RCSSEnv 写入此球员的动作
        self._action_events[unum].wait()
        self._action_events[unum].clear()

        with self._lock:
            actions = self._actions.pop(unum, None)
        return actions if actions is not None else pb2.PlayerActions()

    def get_state(self, unum: int) -> pb2.State | None:
        with self._lock:
            return self._states.get(unum)

    def set_actions(self, unum: int, actions: pb2.PlayerActions) -> None:
        with self._lock:
            self._actions[unum] = actions
        self._action_events[unum].set()
```

### 5.2 单步同步时序

所有 sidecar 连接到同一个 gRPC 端口，`GameServicer` 通过 unum 分发：

```
   Sidecar[unum=2]       Sidecar[unum=3]         GameServicer          RCSSEnv
        │                      │                      │                    │
        │  GetPlayerActions    │  GetPlayerActions     │                    │
        │─────────────────────▶│─────────────────────▶│                    │
        │  (wm.self.id=2)      │  (wm.self.id=3)      │                    │
        │                      │                      │  state_events[2].set()
        │                      │                      │  state_events[3].set()
        │                      │                      │──────────────────▶│
        │                      │                      │                    │
        │                      │                      │  (等待所有 unum 的  │
        │                      │                      │   state_events)    │
        │                      │                      │                    │
        │                      │                      │  (提取 obs, 交 RL)  │
        │                      │                      │  (RL 产出 actions)  │
        │                      │                      │                    │
        │                      │                      │  set_actions(2,...)│
        │                      │                      │  set_actions(3,...)│
        │                      │                      │◀──────────────────│
        │  PlayerActions       │  PlayerActions       │                    │
        │◀─────────────────────│◀─────────────────────│                    │
        │                      │                      │                    │
```

### 5.3 状态同步等待

```python
def _wait_for_states(self, timeout: float = 30.0):
    """等待所有训练 agent 的 sidecar 发送状态"""
    for unum in self._training_unums:
        if not self._servicer.wait_for_state(unum, timeout=timeout):
            raise TimeoutError(
                f"Timeout waiting for state from player unum={unum}"
            )
```

对应 `GameServicer` 需要暴露一个公共方法：

```python
class GameServicer:
    # ...
    def wait_for_state(self, unum: int, timeout: float = 30.0) -> bool:
        """等待指定球员的状态到达，返回是否成功"""
        event = self._state_events.get(unum)
        if event is None:
            raise RuntimeError(f"Player unum={unum} not registered")
        result = event.wait(timeout=timeout)
        if result:
            event.clear()
        return result
```

## 6. PlayerConfig 与 RoomRequest 改进

### 6.1 PlayerConfig 包含完整 template.json 字段

`PlayerConfig` 需要能表达 template.json 中每个球员的完整配置，包括 per-player 的 gRPC 地址（用于 agent）、初始状态和动作黑名单：

```python
@dataclass
class PlayerInitState:
    """per-player 初始状态覆盖"""
    pos_x: float | None = None     # 归一化 X 坐标 (0~1)
    pos_y: float | None = None     # 归一化 Y 坐标 (0~1)
    stamina: float | None = None   # 初始体力

@dataclass
class PlayerConfig:
    unum: int = 1
    goalie: bool = False
    policy_kind: str = "agent"        # "bot" | "agent"
    policy_image: str | None = None   # Docker 镜像 (bot 和 agent 均使用)
    policy_agent: str | None = None   # agent 类型 (如 "ssp"), 仅 kind="agent"
    grpc_host: str | None = None      # gRPC 回连地址, 仅 kind="agent"
    grpc_port: int | None = None      # gRPC 回连端口, 仅 kind="agent"
    init_state: PlayerInitState | None = None  # 可选初始状态
    blocklist: dict[str, bool] | None = None   # 可选动作黑名单
```

> **说明**：gRPC 配置在**每个 agent 球员的 policy 内**分别指定，这与 template.json 的格式一致。
> 虽然通常所有 agent 指向同一个 gRPC 服务器（地址和端口相同），但结构上它们是 per-player 的。

### 6.2 RoomRequest.to_dict() 输出完整 template.json

```python
@dataclass
class RoomRequest:
    ally_name: str = "RLAgent"
    opponent_name: str = "Bot"
    ally_players: list[PlayerConfig] = field(default_factory=list)
    opponent_players: list[PlayerConfig] = field(default_factory=list)
    time_up: int = 6000
    goal_limit_l: int = 0              # stopping.goal_l
    referee_enable: bool = False       # referee.enable
    ball_init_x: float | None = None   # init_state.ball.x
    ball_init_y: float | None = None   # init_state.ball.y

def to_dict(self) -> dict:
    def _player(p: PlayerConfig) -> dict:
        policy = {"kind": p.policy_kind}
        if p.policy_image:
            policy["image"] = p.policy_image
        if p.policy_kind == "agent":
            if p.policy_agent:
                policy["agent"] = p.policy_agent
            if p.grpc_host is not None:
                policy["grpc_host"] = p.grpc_host
            if p.grpc_port is not None:
                policy["grpc_port"] = p.grpc_port
        result = {
            "unum": p.unum,
            "goalie": p.goalie,
            "policy": policy,
        }
        if p.init_state is not None:
            init_dict = p.init_state.to_dict()
            if init_dict:
                result["init_state"] = init_dict
        if p.blocklist:
            result["blocklist"] = p.blocklist
        return result

    result = {
        "api_version": 1,
        "referee": {"enable": self.referee_enable},
        "stopping": {"time_up": self.time_up, "goal_l": self.goal_limit_l},
        "teams": {
            "allies": {
                "name": self.ally_name,
                "players": [_player(p) for p in self.ally_players],
            },
            "opponents": {
                "name": self.opponent_name,
                "players": [_player(p) for p in self.opponent_players],
            },
        },
    }
    if self.ball_init_x is not None and self.ball_init_y is not None:
        result["init_state"] = {
            "ball": {"x": self.ball_init_x, "y": self.ball_init_y},
        }
    return result
```

## 7. 错误处理与鲁棒性

### 7.1 gRPC 连接断开

```python
# 在 GameServicer.GetPlayerActions 中添加超时
def GetPlayerActions(self, request, context):
    unum = request.world_model.self.id
    with self._lock:
        self._states[unum] = request
    self._state_events[unum].set()

    # 添加超时保护
    if not self._action_events[unum].wait(timeout=30.0):
        logger.warning("Timeout waiting for actions for unum=%d, returning empty", unum)
        return pb2.PlayerActions()

    self._action_events[unum].clear()
    with self._lock:
        actions = self._actions.pop(unum, None)
    return actions if actions is not None else pb2.PlayerActions()
```

### 7.2 房间分配失败

```python
def _reset_remote(self, **kwargs):
    try:
        response = self._allocator.request_room(room_request)
        self._room_id = response["room_id"]
    except RuntimeError as e:
        logger.error("Failed to allocate room: %s", e)
        raise gymnasium.error.ResetNeeded("Cannot allocate simulation room") from e
```

### 7.3 优雅关闭

```python
def close(self):
    """确保资源被正确释放"""
    if self._cfg.mode == "remote":
        # 先解除所有阻塞的 action_events
        if self._servicer is not None:
            for unum in self._training_unums:
                self._servicer.set_actions(unum, pb2.PlayerActions())

        # 释放房间
        if self._room_id:
            try:
                self._allocator.release_room(self._room_id)
            except Exception as e:
                logger.warning("Failed to release room: %s", e)
            self._room_id = None

        # 停止 gRPC 服务器
        self._stop_grpc_server()
```

## 8. 与 Ray/RLlib 的集成

### 8.1 多环境并行

Ray/RLlib 的 `num_env_runners` 和 `num_envs_per_runner` 会创建多个 `RCSSEnv` 实例。每个实例只需一个 gRPC 端口（而非 N 个），因此端口分配大幅简化：

```python
MAX_ENVS_PER_WORKER = 10  # 每个 worker 最多的向量化环境数

def __init__(self, config):
    # 使用 worker_index 和 vector_index 分配唯一端口
    worker_index = config.get("worker_index", 0)
    vector_index = config.get("vector_index", 0)
    # 每个 env 实例只需 1 个端口
    self._grpc_port = self._cfg.grpc_port + worker_index * MAX_ENVS_PER_WORKER + vector_index
```

### 8.2 Agent ID 策略映射

```python
# 在 train.py 中：
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # 所有训练 agent 可以共享一个策略（parameter sharing）
    return "shared_policy"

# 或每个 agent 独立策略
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id
```

## 9. 实施计划

### 阶段 1：核心联动
1. 扩展 `EnvConfig` 添加远程模式配置
2. 改造 `RCSSEnv` 支持 `mode="remote"`
3. 实现 gRPC 服务器的启动/停止管理
4. 实现 `RoomRequest` 的 gRPC 地址传递
5. 实现 `reset_remote()` 和 `step_remote()` 的基本流程

### 阶段 2：观测与动作
6. 实现 `WorldModel → observation` 的转换
7. 实现 `action → PlayerActions` 的转换
8. 实现奖励函数
9. 实现终止条件检测

### 阶段 3：鲁棒性
10. 添加超时和错误处理
11. 实现优雅关闭
12. 支持多环境并行（端口分配）
13. 添加集成测试

### 阶段 4：优化
14. 优化观测空间（特征选择、归一化）
15. 支持高级动作空间
16. 添加课程学习支持（curriculum learning）
17. 性能优化（减少 proto 转换开销）
