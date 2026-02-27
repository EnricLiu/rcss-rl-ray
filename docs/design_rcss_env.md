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
              │   Allocator (K8s)     │  │ (Sidecar)             │
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

### 3.1 多智能体 gRPC 架构

**关键理解**：在 rcss 中，每个训练 agent（球员）对应一个独立的 SoccerSimulationProxy sidecar 实例。每个 sidecar 需要连接到一个独立的 gRPC 服务器端口。因此：

- **每个训练 agent 需要一个独立的 `GameServicer` 实例和一个独立的 gRPC 端口**
- 如果配置了 N 个训练 agent，`RCSSEnv` 需要启动 N 个 gRPC 服务器
- 配置房间时，需要在 `template.json` 中为每个训练 agent 提供对应的 gRPC 地址和端口

```
RCSSEnv
  ├── GameServicer[0] (port=50051) ◀──▶ Sidecar[0] (player unum=2)
  ├── GameServicer[1] (port=50052) ◀──▶ Sidecar[1] (player unum=3)
  ├── GameServicer[2] (port=50053) ◀──▶ Sidecar[2] (player unum=7)
  └── ...
```

### 3.2 完整生命周期

```
reset()                                         step()
  │                                               │
  ▼                                               ▼
┌──────────────────────┐                  ┌──────────────────────┐
│ 1. 为每个训练agent    │                  │ 1. 等待所有 servicer  │
│    启动gRPC服务器     │                  │    收到新 State       │
│    (GameServicer)    │                  │    (state_ready.wait) │
│                      │                  │                       │
│ 2. 构建 RoomRequest  │                  │ 2. 从各 WorldModel    │
│    (含gRPC地址列表)   │                  │    提取观测            │
│                      │                  │    (observation)       │
│ 3. AllocatorClient   │                  │                       │
│    .request_room()   │                  │ 3. 返回 obs 给 RLlib  │
│    → 获得 room_id    │                  │    RL产出 action_dict  │
│                      │                  │                       │
│ 4. 等待 sidecar 初始 │                  │ 4. 将 action 转换为   │
│    化完成 (首个State) │                  │    PlayerActions 并   │
│                      │                  │    写入各 servicer     │
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
│ 2. 停止所有 gRPC     │
│    服务器            │
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
    grpc_base_port: int = 50051              # 第一个 gRPC 端口 (递增分配)
    grpc_advertise_host: str = "localhost"   # 告知 sidecar 的回连地址

    # --- 新增：队伍构成 ---
    left_agents: list[int] = field(default_factory=lambda: [2, 3, 4])
    # 左方参与训练的球员 uniform numbers
    right_agents: list[int] = field(default_factory=list)
    # 右方参与训练的球员 uniform numbers (通常为空)
    left_bot_image: str = "helios-base:latest"   # 左方 bot 的 Docker 镜像
    right_bot_image: str = "helios-base:latest"  # 右方 bot 的 Docker 镜像

    # --- 新增：停止条件 ---
    # time_up 控制 rcssserver 的模拟 cycle 上限，由 allocator 传给模拟器
    # max_episode_steps 控制训练端的 episode 截断，可 <= time_up
    time_up: int = 6000                      # rcssserver 模拟 cycle 上限
```

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
        # 为每个训练 agent 创建一个 GameServicer
        self._servicers: dict[str, GameServicer] = {}
        self._grpc_servers: list[grpc.Server] = []
        self._allocator = AllocatorClient(self._cfg.allocator_url)
        self._room_id: str | None = None

    def reset(self, *, seed=None, options=None):
        if self._cfg.mode == "remote":
            return self._reset_remote(seed=seed, options=options)
        return self._reset_local(seed=seed, options=options)

    def _reset_remote(self, *, seed, options):
        # 1. 如果有旧房间，先释放
        self._cleanup_room()

        # 2. 启动 gRPC 服务器
        self._start_grpc_servers()

        # 3. 构建 RoomRequest 并请求房间
        room_request = self._build_room_request()
        response = self._allocator.request_room(room_request)
        self._room_id = response["room_id"]

        # 4. 等待所有 sidecar 完成初始化 (首次收到 State)
        self._wait_for_initial_states()

        # 5. 从各 servicer 提取初始观测
        return self._collect_observations(), self._collect_infos()

    def step(self, action_dict):
        if self._cfg.mode == "remote":
            return self._step_remote(action_dict)
        return self._step_local(action_dict)

    def _step_remote(self, action_dict):
        # 1. 将 action_dict 转换为 PlayerActions 并发送给各 servicer
        for agent_id, action in action_dict.items():
            servicer = self._servicers[agent_id]
            player_actions = self._action_to_proto(action)
            servicer.set_actions(player_actions)

        # 2. 等待所有 servicer 收到下一个 State
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
            self._stop_grpc_servers()
```

### 4.3 gRPC 服务器管理

```python
def _start_grpc_servers(self):
    """为每个训练 agent 启动独立的 gRPC 服务器"""
    port = self._cfg.grpc_base_port

    for agent_id in self._training_agent_ids:
        servicer = GameServicer()
        server = serve(servicer, port=port, block=False)
        self._servicers[agent_id] = servicer
        self._grpc_servers.append(server)
        self._agent_port_map[agent_id] = port
        port += 1

def _stop_grpc_servers(self):
    """停止所有 gRPC 服务器"""
    for server in self._grpc_servers:
        server.stop(grace=5)
    self._grpc_servers.clear()
    self._servicers.clear()
```

### 4.4 RoomRequest 构建

`RoomRequest` 需要根据环境配置生成符合 `template.json` 格式的请求：

```python
def _build_room_request(self) -> RoomRequest:
    """根据配置构建房间请求"""
    left_players = []
    right_players = []

    # 左方队伍
    for unum in range(1, self._cfg.num_left + 1):
        if unum in self._cfg.left_agents:
            # 训练 agent — 需要 gRPC 配置
            agent_id = self._unum_to_agent_id("left", unum)
            left_players.append(PlayerConfig(
                unum=unum,
                goalie=(unum == 1),
                policy_kind="agent",
                grpc_host=self._cfg.grpc_advertise_host,
                grpc_port=self._agent_port_map[agent_id],
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
        # 注：每个 agent 的 grpc_host/port 已在上方的 PlayerConfig 中
        # 按 _agent_port_map 逐个设置，此处字段仅作为全局默认值
        grpc_host=self._cfg.grpc_advertise_host,
        grpc_port=self._cfg.grpc_base_port,
    )
```

### 4.5 template.json 配置映射

allocator 期望的 `template.json` 结构（推测）：

```json
{
  "api_version": 1,
  "stopping": {
    "time_up": 6000
  },
  "teams": {
    "allies": {
      "name": "RLAgent",
      "players": [
        {
          "unum": 1,
          "goalie": true,
          "policy": { "kind": "bot", "image": "helios-base:latest" }
        },
        {
          "unum": 2,
          "goalie": false,
          "policy": {
            "kind": "agent",
            "grpc": { "host": "10.0.0.5", "port": 50051 }
          }
        },
        {
          "unum": 3,
          "goalie": false,
          "policy": {
            "kind": "agent",
            "grpc": { "host": "10.0.0.5", "port": 50052 }
          }
        }
      ]
    },
    "opponents": {
      "name": "Opponent",
      "players": [
        {
          "unum": 1,
          "goalie": true,
          "policy": { "kind": "bot", "image": "helios-base:latest" }
        }
      ]
    }
  }
}
```

**关键点**：每个 `kind: "agent"` 的球员需要附带 `grpc.host` 和 `grpc.port`，告知 sidecar 应该连接到哪里。因此 `RoomRequest.to_dict()` 需要更新以支持每个 agent 的 gRPC 配置。

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

    # 从任一 servicer 获取世界模型
    wm = next(iter(self._servicers.values())).get_world_model()

    # 比赛结束 (game_mode_type == TimeOver)
    game_over = (wm.game_mode_type == pb2.TimeOver)

    # 或达到最大步数
    truncated = (wm.cycle >= self._cfg.max_episode_steps)

    for agent_id in self._training_agent_ids:
        terminateds[agent_id] = game_over
        truncateds[agent_id] = truncated

    terminateds["__all__"] = game_over
    truncateds["__all__"] = truncated

    return terminateds, truncateds
```

## 5. 同步机制

### 5.1 单步同步时序

```
           Sidecar[0]        GameServicer[0]       RCSSEnv         GameServicer[1]        Sidecar[1]
              │                    │                    │                   │                    │
              │  GetPlayerActions  │                    │                   │  GetPlayerActions  │
              │───────────────────▶│                    │                   │◀──────────────────│
              │                    │  state_ready.set() │                   │  state_ready.set() │
              │                    │───────────────────▶│◀──────────────────│                    │
              │                    │                    │                   │                    │
              │                    │         (等待所有 state_ready)          │                    │
              │                    │                    │                   │                    │
              │                    │         (提取 obs, 交给 RL)            │                    │
              │                    │         (RL 产出 actions)             │                    │
              │                    │                    │                   │                    │
              │                    │  action_ready.set()│  action_ready.set()                   │
              │                    │◀───────────────────│──────────────────▶│                    │
              │  PlayerActions     │                    │                   │   PlayerActions    │
              │◀───────────────────│                    │                   │──────────────────▶│
              │                    │                    │                   │                    │
```

### 5.2 状态同步等待

```python
def _wait_for_states(self, timeout: float = 30.0):
    """等待所有训练 agent 的 sidecar 发送状态"""
    for agent_id, servicer in self._servicers.items():
        if not servicer.state_ready.wait(timeout=timeout):
            raise TimeoutError(
                f"Timeout waiting for state from agent {agent_id}"
            )
        servicer.state_ready.clear()
```

## 6. PlayerConfig 与 RoomRequest 改进

### 6.1 PlayerConfig 需要支持 gRPC 配置

```python
@dataclass
class PlayerConfig:
    unum: int = 1
    goalie: bool = False
    policy_kind: str = "agent"      # "bot" | "agent"
    policy_image: str | None = None  # Docker image for bot
    grpc_host: str | None = None     # gRPC host for agent
    grpc_port: int | None = None     # gRPC port for agent
```

### 6.2 RoomRequest.to_dict() 需要包含 gRPC 信息

```python
def to_dict(self) -> dict:
    def _player(p: PlayerConfig) -> dict:
        policy = {"kind": p.policy_kind}
        if p.policy_kind == "bot" and p.policy_image:
            policy["image"] = p.policy_image
        if p.policy_kind == "agent" and p.grpc_host:
            policy["grpc"] = {
                "host": p.grpc_host,
                "port": p.grpc_port,
            }
        return {
            "unum": p.unum,
            "goalie": p.goalie,
            "policy": policy,
        }
    # ...
```

## 7. 错误处理与鲁棒性

### 7.1 gRPC 连接断开

```python
# 在 GameServicer.GetPlayerActions 中添加超时
def GetPlayerActions(self, request, context):
    with self._lock:
        self._current_state = request
    self.state_ready.set()

    # 添加超时保护
    if not self.action_ready.wait(timeout=30.0):
        logger.warning("Timeout waiting for actions, returning empty")
        return pb2.PlayerActions()

    self.action_ready.clear()
    with self._lock:
        actions = self._current_actions
        self._current_actions = None
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
        # 先解除所有阻塞的 servicer
        for servicer in self._servicers.values():
            servicer.set_actions(pb2.PlayerActions())  # 解除阻塞

        # 释放房间
        if self._room_id:
            try:
                self._allocator.release_room(self._room_id)
            except Exception as e:
                logger.warning("Failed to release room: %s", e)
            self._room_id = None

        # 停止 gRPC 服务器
        self._stop_grpc_servers()
```

## 8. 与 Ray/RLlib 的集成

### 8.1 多环境并行

Ray/RLlib 的 `num_env_runners` 和 `num_envs_per_runner` 会创建多个 `RCSSEnv` 实例。每个实例需要：
- 独立的 gRPC 端口范围（避免冲突）
- 独立的模拟房间

```python
# 每个 (worker, vector_env) 组合需要 n_agents 个端口。
# PORT_BLOCK_SIZE 定义每个 env 实例保留的端口数量（应 >= 最大训练 agent 数）。
PORT_BLOCK_SIZE = 100  # 保留 100 个端口 per env 实例，允许最多 100 个训练 agent

def __init__(self, config):
    # 使用 worker_index 和 vector_index 来避免端口冲突
    worker_index = config.get("worker_index", 0)
    vector_index = config.get("vector_index", 0)
    n_agents = len(self._training_agent_ids)
    base_port = self._cfg.grpc_base_port + (worker_index * PORT_BLOCK_SIZE + vector_index) * n_agents
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
