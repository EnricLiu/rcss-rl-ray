"""Multi-agent RCSS gymnasium environment.

This module implements :class:`RCSSEnv`, a multi-agent reinforcement-learning
environment for RoboCup Soccer Simulation.  The environment follows the
:mod:`gymnasium` API and is compatible with Ray/RLlib multi-agent training.

Two operating modes are supported:

``mode="local"`` (default)
    A built-in simplified simulation is used.  No external services are
    required; useful for smoke-testing and algorithm prototyping.

``mode="remote"``
    The environment contacts the *rcss_cluster* allocator via REST to spin
    up a full simulation room, then communicates with
    SoccerSimulationProxy sidecars over gRPC for each training agent.
    See :mod:`rcss_rl.env.allocator_client` and
    :mod:`rcss_rl.env.grpc_service`.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from rcss_rl.config import EnvConfig, PlayerConfig, PlayerInitState

logger = logging.getLogger(__name__)

# Max vectorised envs per Ray worker – used for gRPC port allocation.
_MAX_ENVS_PER_WORKER: int = 10


# ---------------------------------------------------------------------------
# Dict → dataclass helpers (needed when RLlib passes config as a dict)
# ---------------------------------------------------------------------------

def _player_config_from_dict(d: dict[str, Any]) -> PlayerConfig:
    """Reconstruct a :class:`PlayerConfig` from a plain dict."""
    d = dict(d)  # shallow copy — original dict is not mutated
    init_state = d.pop("init_state", None)  # extract before **d unpacking
    if isinstance(init_state, dict):
        init_state = PlayerInitState(**init_state)
    return PlayerConfig(**d, init_state=init_state)


def _env_config_from_dict(d: dict[str, Any]) -> EnvConfig:
    """Reconstruct an :class:`EnvConfig` from a (possibly nested) dict.

    Handles the ``ally_players`` / ``opponent_players`` lists whose
    elements may be plain dicts (as produced by ``dataclasses.asdict``).
    The original dict *d* is not mutated.
    """
    d = dict(d)  # shallow copy — original dict is not mutated
    for key in ("ally_players", "opponent_players"):
        raw = d.get(key)
        if raw is not None:
            d[key] = [
                _player_config_from_dict(p) if isinstance(p, dict) else p
                for p in raw
            ]
    return EnvConfig(**d)


# ---------------------------------------------------------------------------
# Discrete action definitions (local mode + remote low-level actions)
# ---------------------------------------------------------------------------
NOOP = 0
DASH_FORWARD = 1
DASH_BACKWARD = 2
TURN_LEFT = 3
TURN_RIGHT = 4
KICK = 5
NUM_ACTIONS = 6

# Turn amount per action (radians, ≈ 30°).
_TURN_ANGLE = np.pi / 6.0

# Velocity decay applied each step (simple friction model).
_VELOCITY_DECAY = 0.4

# Field half-width ratio (approximation of the standard 105×68 pitch).
_FIELD_WIDTH_RATIO = 0.68


class RCSSEnv(gymnasium.Env):
    """Multi-agent RCSS environment supporting local and remote modes.

    **Local mode** — built-in simplified simulation (default).

    Each agent observes:

    * Ball position & velocity (4)
    * Self position & velocity (4)
    * Positions of all other agents ``(n_agents - 1) * 2``
    * Left / right team scores (2)
    * Normalised step count (1)
    * Ball-kickable flag (1)

    Total observation dimension = ``8 + (num_left + num_right - 1) * 2 + 4``.

    **Remote mode** — connects to rcss_cluster via REST / gRPC.

    The environment allocates a simulation room, starts a gRPC server,
    and exchanges ``WorldModel`` / ``PlayerActions`` messages with
    SoccerSimulationProxy sidecars.

    Parameters
    ----------
    config:
        An :class:`~rcss_rl.config.EnvConfig`, a plain ``dict`` whose keys
        match ``EnvConfig`` fields, or *None* for default settings.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: EnvConfig | dict[str, Any] | None = None) -> None:
        super().__init__()

        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            config = _env_config_from_dict(config)

        self._cfg: EnvConfig = config
        n_agents = config.num_left + config.num_right

        # Identify which players are RL-training agents (kind="agent").
        self._ally_agent_unums: list[int] = [
            p.unum for p in config.ally_players if p.policy_kind == "agent"
        ]
        self._opponent_agent_unums: list[int] = [
            p.unum for p in config.opponent_players if p.policy_kind == "agent"
        ]

        if config.mode == "local":
            # Local mode: ALL players are controllable; use position-based IDs.
            self._agent_id_list: list[str] = [
                f"left_{i}" for i in range(config.num_left)
            ] + [f"right_{i}" for i in range(config.num_right)]
        else:
            # Remote mode: only RL agents; use unum-based IDs.
            self._agent_id_list = [
                f"left_{u}" for u in self._ally_agent_unums
            ] + [f"right_{u}" for u in self._opponent_agent_unums]
        self._agent_ids: set[str] = set(self._agent_id_list)

        # For the *local* mode we also need identifiers for ALL players
        # (agents + bots) to drive the simplified simulation.
        self._all_id_list: list[str] = [
            f"left_{i}" for i in range(config.num_left)
        ] + [f"right_{i}" for i in range(config.num_right)]

        # Spaces
        self._obs_dim = 8 + (n_agents - 1) * 2 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Mutable state (populated by :meth:`reset`).
        self._step_count: int = 0
        self._rng: np.random.Generator | None = None
        self._agent_pos: dict[str, np.ndarray] = {}
        self._agent_vel: dict[str, np.ndarray] = {}
        self._agent_dir: dict[str, float] = {}
        self._ball_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._ball_vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self._left_score: int = 0
        self._right_score: int = 0

        # Remote-mode components (lazily initialised).
        self._servicer: Any = None
        self._grpc_server: Any = None
        self._allocator: Any = None
        self._room_id: str | None = None
        self._prev_world_models: dict[int, Any] = {}

        if config.mode == "remote":
            self._setup_remote()

    # ------------------------------------------------------------------
    # Remote-mode setup
    # ------------------------------------------------------------------

    def _setup_remote(self) -> None:
        """Initialise components needed for remote mode."""
        from rcss_rl.env.allocator_client import AllocatorClient
        from rcss_rl.env.grpc_service import GameServicer

        self._servicer = GameServicer()

        # Register every training-agent unum with the servicer.
        for u in self._ally_agent_unums:
            self._servicer.register_player(u)
        for u in self._opponent_agent_unums:
            self._servicer.register_player(u)

        self._allocator = AllocatorClient(self._cfg.allocator_url)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the environment and return initial observations."""
        if self._cfg.mode == "remote":
            return self._reset_remote(seed=seed, options=options)
        return self._reset_local(seed=seed, options=options)

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Execute one simulation step."""
        if self._cfg.mode == "remote":
            return self._step_remote(action_dict)
        return self._step_local(action_dict)

    def close(self) -> None:
        """Release external resources (remote mode)."""
        if self._cfg.mode == "remote":
            self._cleanup_room()
            self._stop_grpc_server()

    # ------------------------------------------------------------------
    # LOCAL mode implementation
    # ------------------------------------------------------------------

    def _reset_local(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(self._cfg.seed)

        self._step_count = 0
        self._left_score = 0
        self._right_score = 0

        self._init_positions()

        obs = {aid: self._get_obs(aid) for aid in self._all_id_list}
        infos: dict[str, dict[str, Any]] = {aid: {} for aid in self._all_id_list}
        return obs, infos

    def _step_local(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        self._step_count += 1
        acting_agents = set(action_dict.keys())

        for aid in self._all_id_list:
            self._apply_action(aid, action_dict.get(aid, NOOP))

        self._update_physics()

        goal_scored = self._check_goals()

        truncated = self._step_count >= self._cfg.max_episode_steps

        obs = {aid: self._get_obs(aid) for aid in acting_agents}
        rewards = {aid: self._compute_reward(aid, goal_scored) for aid in acting_agents}

        terminateds: dict[str, bool] = {aid: False for aid in acting_agents}
        terminateds["__all__"] = False

        truncateds: dict[str, bool] = {aid: truncated for aid in acting_agents}
        truncateds["__all__"] = truncated

        infos: dict[str, dict[str, Any]] = {
            aid: {
                "scores": {"left": self._left_score, "right": self._right_score},
                "step": self._step_count,
            }
            for aid in acting_agents
        }

        return obs, rewards, terminateds, truncateds, infos

    # ---- Local simulation helpers ----------------------------------------

    def _init_positions(self) -> None:
        """Randomise starting positions of agents and the ball."""
        assert self._rng is not None
        half_x = self._cfg.field_half_x
        half_y = half_x * _FIELD_WIDTH_RATIO

        self._ball_pos = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)

        for i in range(self._cfg.num_left):
            aid = f"left_{i}"
            x = float(self._rng.uniform(-half_x, 0))
            y = float(self._rng.uniform(-half_y, half_y))
            self._agent_pos[aid] = np.array([x, y], dtype=np.float32)
            self._agent_vel[aid] = np.zeros(2, dtype=np.float32)
            self._agent_dir[aid] = 0.0

        for i in range(self._cfg.num_right):
            aid = f"right_{i}"
            x = float(self._rng.uniform(0, half_x))
            y = float(self._rng.uniform(-half_y, half_y))
            self._agent_pos[aid] = np.array([x, y], dtype=np.float32)
            self._agent_vel[aid] = np.zeros(2, dtype=np.float32)
            self._agent_dir[aid] = np.pi

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Construct the observation vector for *agent_id*."""
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        idx = 0

        obs[idx : idx + 2] = self._ball_pos; idx += 2
        obs[idx : idx + 2] = self._ball_vel; idx += 2
        obs[idx : idx + 2] = self._agent_pos[agent_id]; idx += 2
        obs[idx : idx + 2] = self._agent_vel[agent_id]; idx += 2

        for aid in self._all_id_list:
            if aid != agent_id:
                obs[idx : idx + 2] = self._agent_pos[aid]
                idx += 2

        obs[idx] = float(self._left_score); idx += 1
        obs[idx] = float(self._right_score); idx += 1
        obs[idx] = self._step_count / max(self._cfg.max_episode_steps, 1); idx += 1

        dist = float(np.linalg.norm(self._ball_pos - self._agent_pos[agent_id]))
        obs[idx] = 1.0 if dist <= self._cfg.kick_radius else 0.0

        return obs

    def _apply_action(self, agent_id: str, action: int) -> None:
        """Translate a discrete action into velocity / direction changes."""
        cfg = self._cfg
        direction = self._agent_dir[agent_id]

        if action == DASH_FORWARD:
            self._agent_vel[agent_id] = np.array(
                [cfg.move_speed * np.cos(direction), cfg.move_speed * np.sin(direction)],
                dtype=np.float32,
            )
        elif action == DASH_BACKWARD:
            self._agent_vel[agent_id] = np.array(
                [-cfg.move_speed * np.cos(direction), -cfg.move_speed * np.sin(direction)],
                dtype=np.float32,
            )
        elif action == TURN_LEFT:
            self._agent_dir[agent_id] -= _TURN_ANGLE
        elif action == TURN_RIGHT:
            self._agent_dir[agent_id] += _TURN_ANGLE
        elif action == KICK:
            dist = float(np.linalg.norm(self._ball_pos - self._agent_pos[agent_id]))
            if dist <= cfg.kick_radius:
                kick_dir = 0.0 if agent_id.startswith("left") else np.pi
                self._ball_vel = np.array(
                    [cfg.kick_power * np.cos(kick_dir), cfg.kick_power * np.sin(kick_dir)],
                    dtype=np.float32,
                )

    def _update_physics(self) -> None:
        """Integrate velocities and apply friction / boundary clamping."""
        half_x = self._cfg.field_half_x
        half_y = half_x * _FIELD_WIDTH_RATIO

        for aid in self._all_id_list:
            self._agent_pos[aid] += self._agent_vel[aid]
            self._agent_vel[aid] *= _VELOCITY_DECAY
            self._agent_pos[aid][0] = np.clip(self._agent_pos[aid][0], -half_x, half_x)
            self._agent_pos[aid][1] = np.clip(self._agent_pos[aid][1], -half_y, half_y)

        self._ball_pos += self._ball_vel
        self._ball_vel *= _VELOCITY_DECAY

    def _check_goals(self) -> str | None:
        """Return ``"left"``/``"right"`` when a goal is scored, else *None*."""
        half_x = self._cfg.field_half_x
        if self._ball_pos[0] > half_x:
            self._left_score += 1
            self._ball_pos[:] = 0.0
            self._ball_vel[:] = 0.0
            return "left"
        if self._ball_pos[0] < -half_x:
            self._right_score += 1
            self._ball_pos[:] = 0.0
            self._ball_vel[:] = 0.0
            return "right"
        return None

    def _compute_reward(self, agent_id: str, goal_scored: str | None) -> float:
        """Compute the scalar reward for *agent_id*."""
        cfg = self._cfg
        reward = 0.0

        if goal_scored is not None:
            team = agent_id.split("_")[0]
            reward += cfg.goal_reward if goal_scored == team else -cfg.goal_reward

        dist = float(np.linalg.norm(self._ball_pos - self._agent_pos[agent_id]))
        reward -= cfg.distance_penalty * dist
        return reward

    # ------------------------------------------------------------------
    # REMOTE mode implementation
    # ------------------------------------------------------------------

    def _reset_remote(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset in remote mode: allocate room and wait for initial states."""
        from rcss_rl.env.allocator_client import RoomRequest

        # 1. Clean up any previous room.
        self._cleanup_room()

        # 2. Start gRPC server (if not already running).
        self._start_grpc_server()

        # 3. Clear servicer buffers for the new episode.
        self._servicer.reset()

        # 4. Build and send room request.
        room_request = self._build_room_request()
        try:
            response = self._allocator.request_room(room_request)
        except RuntimeError as exc:
            raise gymnasium.error.ResetNeeded(
                "Cannot allocate simulation room"
            ) from exc
        self._room_id = response.get("room_id")

        # 5. Wait for all training agents' sidecars to send initial state.
        self._wait_for_all_states()

        # 6. Collect initial observations and infos.
        self._step_count = 0
        self._prev_world_models = self._collect_world_models()
        obs = self._collect_remote_observations()
        infos: dict[str, dict[str, Any]] = {aid: {} for aid in self._agent_id_list}
        return obs, infos

    def _step_remote(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Execute one remote step: send actions, wait for next states."""
        self._step_count += 1

        # 1. Convert each agent's action to proto and push to servicer.
        for agent_id, action in action_dict.items():
            side, unum = self._parse_agent_id(agent_id)
            proto_actions = self._action_to_proto(action)
            self._servicer.set_actions(unum, proto_actions)

        # Send NOOP for agents not present in action_dict.
        for agent_id in self._agent_id_list:
            if agent_id not in action_dict:
                _, unum = self._parse_agent_id(agent_id)
                self._servicer.set_actions(unum, self._action_to_proto(NOOP))

        # 2. Wait for next states from all training agents.
        self._wait_for_all_states()

        # 3. Collect observations, rewards, termination, truncation.
        prev_wms = self._prev_world_models
        curr_wms = self._collect_world_models()
        self._prev_world_models = curr_wms

        acting_agents = set(action_dict.keys())

        obs = self._collect_remote_observations()
        rewards = {
            aid: self._compute_remote_reward(aid, prev_wms, curr_wms)
            for aid in acting_agents
        }
        terminateds, truncateds = self._check_remote_done(curr_wms)
        infos = self._collect_remote_infos(curr_wms, acting_agents)

        return obs, rewards, terminateds, truncateds, infos

    # ---- Remote helpers --------------------------------------------------

    def _build_room_request(self) -> Any:
        """Build a :class:`RoomRequest` from the current :class:`EnvConfig`."""
        from rcss_rl.env.allocator_client import RoomRequest

        return RoomRequest(
            ally_name=self._cfg.ally_team_name,
            opponent_name=self._cfg.opponent_team_name,
            ally_players=self._cfg.ally_players,
            opponent_players=self._cfg.opponent_players,
            time_up=self._cfg.time_up,
            goal_limit_l=self._cfg.goal_limit_l,
            referee_enable=self._cfg.referee_enable,
            ball_init_x=self._cfg.ball_init_x,
            ball_init_y=self._cfg.ball_init_y,
        )

    def _start_grpc_server(self) -> None:
        """Start the gRPC server if not already running."""
        if self._grpc_server is not None:
            return
        from rcss_rl.env.grpc_service import serve

        self._grpc_server = serve(
            self._servicer,
            port=self._cfg.grpc_port,
            block=False,
        )

    def _stop_grpc_server(self) -> None:
        """Stop the gRPC server."""
        if self._grpc_server is not None:
            self._grpc_server.stop(grace=5)
            self._grpc_server = None

    def _cleanup_room(self) -> None:
        """Release any previously allocated room."""
        if self._room_id and self._allocator is not None:
            try:
                self._allocator.release_room(self._room_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to release room %s: %s", self._room_id, exc)
            self._room_id = None

    def _wait_for_all_states(self, timeout: float = 30.0) -> None:
        """Wait for every registered training agent to send a state."""
        for unum in list(self._ally_agent_unums) + list(self._opponent_agent_unums):
            if not self._servicer.wait_for_state(unum, timeout=timeout):
                raise TimeoutError(
                    f"Timeout waiting for state from player unum={unum}"
                )

    def _collect_world_models(self) -> dict[int, Any]:
        """Collect current world models from the servicer, keyed by unum."""
        wms: dict[int, Any] = {}
        for unum in list(self._ally_agent_unums) + list(self._opponent_agent_unums):
            wms[unum] = self._servicer.get_world_model(unum)
        return wms

    def _collect_remote_observations(self) -> dict[str, np.ndarray]:
        """Build observation vectors from world models for all training agents."""
        obs: dict[str, np.ndarray] = {}
        for agent_id in self._agent_id_list:
            _, unum = self._parse_agent_id(agent_id)
            wm = self._servicer.get_world_model(unum)
            obs[agent_id] = self._world_model_to_obs(wm)
        return obs

    def _world_model_to_obs(self, wm: Any) -> np.ndarray:
        """Convert a protobuf ``WorldModel`` to a numpy observation vector.

        The observation layout:

        * Ball pos/vel (4)
        * Self pos/vel (4)
        * Self body_direction, stamina, is_kickable (3)
        * Teammates pos/vel sorted by unum (num_teammates * 4)
        * Opponents pos/vel sorted by unum (num_opponents * 4)
        * Cycle, game_mode, our_score, their_score (4)
        """
        if wm is None:
            return np.zeros(self._obs_dim, dtype=np.float32)

        obs_parts: list[float] = []

        # Ball
        ball = wm.ball
        obs_parts.extend([
            ball.position.x, ball.position.y,
            ball.velocity.x, ball.velocity.y,
        ])

        # Self
        s = wm.self
        obs_parts.extend([
            s.position.x, s.position.y,
            s.velocity.x, s.velocity.y,
            s.body_direction, s.stamina,
            float(s.is_kickable),
        ])

        # Teammates sorted by uniform_number
        teammates = sorted(wm.teammates, key=lambda p: p.uniform_number)
        for tm in teammates:
            obs_parts.extend([
                tm.position.x, tm.position.y,
                tm.velocity.x, tm.velocity.y,
            ])

        # Opponents sorted by uniform_number
        opponents = sorted(wm.opponents, key=lambda p: p.uniform_number)
        for opp in opponents:
            obs_parts.extend([
                opp.position.x, opp.position.y,
                opp.velocity.x, opp.velocity.y,
            ])

        # Match state
        obs_parts.extend([
            float(wm.cycle),
            float(wm.game_mode_type),
            float(wm.our_team_score),
            float(wm.their_team_score),
        ])

        arr = np.array(obs_parts, dtype=np.float32)
        # Pad or truncate to obs_dim for safety.
        result = np.zeros(self._obs_dim, dtype=np.float32)
        n = min(len(arr), self._obs_dim)
        result[:n] = arr[:n]
        return result

    def _action_to_proto(self, action: int) -> Any:
        """Convert a discrete action integer to a protobuf ``PlayerActions``."""
        from rcss_rl.proto import service_pb2 as pb2

        pa = pb2.PlayerActions()

        if action == DASH_FORWARD:
            pa.actions.append(pb2.PlayerAction(
                dash=pb2.Dash(power=100.0, relative_direction=0.0),
            ))
        elif action == DASH_BACKWARD:
            pa.actions.append(pb2.PlayerAction(
                dash=pb2.Dash(power=-100.0, relative_direction=0.0),
            ))
        elif action == TURN_LEFT:
            pa.actions.append(pb2.PlayerAction(
                turn=pb2.Turn(relative_direction=-30.0),
            ))
        elif action == TURN_RIGHT:
            pa.actions.append(pb2.PlayerAction(
                turn=pb2.Turn(relative_direction=30.0),
            ))
        elif action == KICK:
            pa.actions.append(pb2.PlayerAction(
                kick=pb2.Kick(power=100.0, relative_direction=0.0),
            ))
        # NOOP → empty actions list.

        # Always add neck scan as a secondary action.
        pa.actions.append(pb2.PlayerAction(
            neck_turn_to_ball_or_scan=pb2.Neck_TurnToBallOrScan(),
        ))

        return pa

    def _compute_remote_reward(
        self,
        agent_id: str,
        prev_wms: dict[int, Any],
        curr_wms: dict[int, Any],
    ) -> float:
        """Compute reward for *agent_id* from consecutive world models."""
        _, unum = self._parse_agent_id(agent_id)
        prev_wm = prev_wms.get(unum)
        curr_wm = curr_wms.get(unum)
        if prev_wm is None or curr_wm is None:
            return 0.0

        reward = 0.0

        # Goal reward.
        score_diff = curr_wm.our_team_score - prev_wm.our_team_score
        opp_diff = curr_wm.their_team_score - prev_wm.their_team_score
        reward += score_diff * self._cfg.goal_reward
        reward -= opp_diff * self._cfg.goal_reward

        # Distance-to-ball penalty.
        reward -= self._cfg.distance_penalty * curr_wm.self.dist_from_ball

        return reward

    def _check_remote_done(
        self, curr_wms: dict[int, Any],
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Determine termination/truncation from world models."""
        from rcss_rl.proto import service_pb2 as pb2

        # Use the first available world model.
        wm = next((w for w in curr_wms.values() if w is not None), None)

        game_over = wm is not None and wm.game_mode_type == pb2.TimeOver
        truncated = self._step_count >= self._cfg.max_episode_steps

        terminateds: dict[str, bool] = {aid: game_over for aid in self._agent_id_list}
        terminateds["__all__"] = game_over

        truncateds: dict[str, bool] = {aid: truncated for aid in self._agent_id_list}
        truncateds["__all__"] = truncated

        return terminateds, truncateds

    def _collect_remote_infos(
        self,
        curr_wms: dict[int, Any],
        acting_agents: set[str],
    ) -> dict[str, dict[str, Any]]:
        """Build per-agent info dicts from world models."""
        infos: dict[str, dict[str, Any]] = {}
        for agent_id in acting_agents:
            _, unum = self._parse_agent_id(agent_id)
            wm = curr_wms.get(unum)
            if wm is not None:
                infos[agent_id] = {
                    "scores": {
                        "left": wm.our_team_score,
                        "right": wm.their_team_score,
                    },
                    "step": self._step_count,
                    "cycle": wm.cycle,
                }
            else:
                infos[agent_id] = {"step": self._step_count}
        return infos

    # ---- Agent ID helpers ------------------------------------------------

    @staticmethod
    def _parse_agent_id(agent_id: str) -> tuple[str, int]:
        """Parse ``"left_3"`` → ``("left", 3)``."""
        side, unum_str = agent_id.split("_", 1)
        return side, int(unum_str)
