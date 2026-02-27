"""Multi-agent RCSS gymnasium environment.

This module implements :class:`RCSSEnv`, a multi-agent reinforcement-learning
environment for RoboCup Soccer Simulation.  The environment follows the
:mod:`gymnasium` API and is compatible with Ray/RLlib multi-agent training.

Currently a **local** simplified simulation is used so that the environment
can be instantiated without any external services.  The architecture is
designed for future integration with:

* **rcss_cluster** – a Kubernetes-hosted match allocator contacted via REST
  to spin up simulation rooms (see :mod:`rcss_rl.env.allocator_client`).
* **SoccerSimulationProxy** – a sidecar that relays world-model states and
  player actions over gRPC (see :mod:`rcss_rl.env.grpc_service`).
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from rcss_rl.config import EnvConfig, PlayerConfig, PlayerInitState

logger = logging.getLogger(__name__)


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
# Discrete action definitions
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
    """Multi-agent RCSS environment with simplified local simulation.

    Each agent observes:

    * Ball position & velocity (4)
    * Self position & velocity (4)
    * Positions of all other agents ``(n_agents - 1) * 2``
    * Left / right team scores (2)
    * Normalised step count (1)
    * Ball-kickable flag (1)

    Total observation dimension = ``8 + (num_left + num_right - 1) * 2 + 4``.

    The discrete action space has six choices:

    0. NOOP
    1. Dash forward
    2. Dash backward
    3. Turn left (−30°)
    4. Turn right (+30°)
    5. Kick ball toward opponent goal

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

        # Agent identifiers (deterministic order: left first, then right).
        self._agent_id_list: list[str] = [
            f"left_{i}" for i in range(config.num_left)
        ] + [f"right_{i}" for i in range(config.num_right)]
        self._agent_ids: set[str] = set(self._agent_id_list)

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

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the environment and return initial observations.

        Parameters
        ----------
        seed:
            If given, re-seeds the internal RNG for reproducible episodes.
        options:
            Currently unused; reserved for future extensions.

        Returns
        -------
        obs:
            Mapping from agent-id to observation array.
        infos:
            Mapping from agent-id to (empty) info dict.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(self._cfg.seed)

        self._step_count = 0
        self._left_score = 0
        self._right_score = 0

        self._init_positions()

        obs = {aid: self._get_obs(aid) for aid in self._agent_id_list}
        infos: dict[str, dict[str, Any]] = {aid: {} for aid in self._agent_id_list}
        return obs, infos

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
        """Execute one simulation step.

        Parameters
        ----------
        action_dict:
            Mapping from agent-id to a discrete action integer.  Agents not
            present in the dict are treated as taking :data:`NOOP`.

        Returns
        -------
        obs:
            Observations for every agent that provided an action.
        rewards:
            Scalar rewards for every agent that provided an action.
        terminateds:
            Per-agent termination flags plus ``"__all__"``.
        truncateds:
            Per-agent truncation flags plus ``"__all__"``.
        infos:
            Per-agent info dicts with ``"scores"`` and ``"step"`` keys.
        """
        self._step_count += 1
        acting_agents = set(action_dict.keys())

        # 1. Apply each agent's action (non-acting agents get NOOP).
        for aid in self._agent_id_list:
            self._apply_action(aid, action_dict.get(aid, NOOP))

        # 2. Physics update.
        self._update_physics()

        # 3. Goal detection.
        goal_scored = self._check_goals()

        # 4. Build return dicts (keyed only by acting agents).
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_positions(self) -> None:
        """Randomise starting positions of agents and the ball."""
        assert self._rng is not None
        half_x = self._cfg.field_half_x
        half_y = half_x * _FIELD_WIDTH_RATIO

        # Ball near centre.
        self._ball_pos = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)

        for i in range(self._cfg.num_left):
            aid = f"left_{i}"
            x = float(self._rng.uniform(-half_x, 0))
            y = float(self._rng.uniform(-half_y, half_y))
            self._agent_pos[aid] = np.array([x, y], dtype=np.float32)
            self._agent_vel[aid] = np.zeros(2, dtype=np.float32)
            self._agent_dir[aid] = 0.0  # facing right

        for i in range(self._cfg.num_right):
            aid = f"right_{i}"
            x = float(self._rng.uniform(0, half_x))
            y = float(self._rng.uniform(-half_y, half_y))
            self._agent_pos[aid] = np.array([x, y], dtype=np.float32)
            self._agent_vel[aid] = np.zeros(2, dtype=np.float32)
            self._agent_dir[aid] = np.pi  # facing left

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Construct the observation vector for *agent_id*."""
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        idx = 0

        # Ball position & velocity (4).
        obs[idx : idx + 2] = self._ball_pos
        idx += 2
        obs[idx : idx + 2] = self._ball_vel
        idx += 2

        # Self position & velocity (4).
        obs[idx : idx + 2] = self._agent_pos[agent_id]
        idx += 2
        obs[idx : idx + 2] = self._agent_vel[agent_id]
        idx += 2

        # Other agents' positions ((n-1)*2).
        for aid in self._agent_id_list:
            if aid != agent_id:
                obs[idx : idx + 2] = self._agent_pos[aid]
                idx += 2

        # Scores (2).
        obs[idx] = float(self._left_score)
        idx += 1
        obs[idx] = float(self._right_score)
        idx += 1

        # Normalised step count (1).
        obs[idx] = self._step_count / max(self._cfg.max_episode_steps, 1)
        idx += 1

        # Ball kickable flag (1).
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

        for aid in self._agent_id_list:
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
            # Ball crossed the right boundary → left team scores.
            self._left_score += 1
            self._ball_pos[:] = 0.0
            self._ball_vel[:] = 0.0
            return "left"
        if self._ball_pos[0] < -half_x:
            # Ball crossed the left boundary → right team scores.
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
