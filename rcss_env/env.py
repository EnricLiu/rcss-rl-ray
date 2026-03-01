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

from schema import RoomSchema, PlayerSchema, PlayerInitState, TeamSide
from config import EnvConfig

import obs as observation
import reward
from .allocator import AllocatorClient
from .grpc_srv import GameServicer, pb2


logger = logging.getLogger(__name__)

# Max vectorized envs per Ray worker – used for gRPC port allocation.
_MAX_ENVS_PER_WORKER: int = 10

def _player_config_from_dict(d: dict[str, Any]) -> PlayerSchema:
    """Reconstruct a :class:`PlayerSchema` from a plain dict."""
    d = dict(d)  # shallow copy — original dict is not mutated
    init_state = d.pop("init_state", None)  # extract before **d unpacking
    if isinstance(init_state, dict):
        init_state = PlayerInitState(**init_state)
    return PlayerSchema(**d, init_state=init_state)


def _env_config_from_dict(d: dict[str, Any]) -> RoomSchema:
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
    return RoomSchema(**d)


NOOP = 0
DASH_FORWARD = 1
DASH_BACKWARD = 2
TURN_LEFT = 3
TURN_RIGHT = 4
KICK = 5
NUM_ACTIONS = 6

class RCSSEnv(gymnasium.Env):
    """Multi-agent RCSS environment

    connects to rcss_cluster via REST / gRPC.

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

    def __init__(self, config: EnvConfig) -> None:
        super().__init__()
        self.__cfg = config

        self.agent_team = self.room_schema.teams.agent_team
        self.agent_team_unums = set([agent.unum for agent in self.agent_team.ssp_agents()])

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Mutable state (populated by :meth:`reset`).
        self._step_count: int = 0

        # Remote-mode components (lazily initialised).
        self.__loop = None

        self.__servicer: GameServicer | None = None
        self.__grpc_server: Any = None
        self.__grpc_loop: Any = None  # asyncio event loop for gRPC aio server

        self.__allocator: Any = None
        self.__room: str | None = None

        self._prev_world_models: dict[int, Any] = {}

        self._setup()

    @property
    def config(self) -> EnvConfig:
        return self.__cfg

    @property
    def room_schema(self) -> RoomSchema:
        """The environment schema (``EnvSchema``) used to configure this env."""
        return self.config.room

    def _setup(self) -> None:
        self.__allocator = AllocatorClient(self.config.allocator.addr)
        self.__servicer = GameServicer()

        for unum in self.agent_team_unums:
            self.__servicer.register(unum)


    @property
    def _agent_side(self) -> str:
        """Return the side string (``"left"`` or ``"right"``) for the agent team."""
        return "left" if self.agent_team.side == TeamSide.LEFT else "right"

    @property
    def _agent_id_list(self) -> list[str]:
        """Sorted list of agent IDs, e.g. ``["left_2", "left_7"]``."""
        side = self._agent_side
        return [f"{side}_{unum}" for unum in sorted(self.agent_team_unums)]

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
        # 1. Clean up any previous room.
        self._cleanup_room()

        # 2. Start gRPC server (if not already running).
        self._start_grpc_server()

        # 3. Clear servicer buffers for the new episode.
        self.__servicer.reset()

        # 4. Build and send room request.
        schema = self.config.room
        try:
            response = self.__allocator.request_room(schema)
        except RuntimeError as exc:
            raise gymnasium.error.ResetNeeded(
                "Cannot allocate simulation room"
            ) from exc
        room_id = response.get("room_id")
        if not room_id:
            raise gymnasium.error.ResetNeeded(
                f"Allocator did not return a valid 'room_id'. Response: {response!r}"
            )
        self.__room = room_id

        # 5. Wait for all training agents' sidecars to send initial state.
        self._wait_for_all_states()

        # 6. Collect initial observations and infos.
        self._step_count = 0
        self._prev_world_models = self._collect_world_models()
        obs = self._collect_observations()
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
        """Execute one simulation step: send actions, wait for next states."""
        self._step_count += 1

        # 1. Build proto actions for every agent and send as a batch.
        proto_actions: dict[int, Any] = {}
        for agent_id in self._agent_id_list:
            _, unum = self._parse_agent_id(agent_id)
            action = action_dict.get(agent_id, NOOP)
            proto_actions[unum] = self._action_to_proto(action)

        self.__servicer.send_actions(proto_actions)

        # 2. Wait for next states from all training agents.
        self._wait_for_all_states()

        # 3. Collect observations, rewards, termination, truncation.
        prev_wms = self._prev_world_models
        curr_wms = self._collect_world_models()
        self._prev_world_models = curr_wms

        obs = self._collect_observations()
        rewards = {
            aid: self._calc_reward(aid, prev_wms, curr_wms)
            for aid in self._agent_id_list
        }
        terminateds, truncateds = self._check_done(curr_wms)
        infos = self._collect_infos(curr_wms)

        return obs, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        """Release external resources."""
        self._cleanup_room()
        self._stop_grpc_server()

    # ------------------------------------------------------------------
    # Room / gRPC lifecycle helpers
    # ------------------------------------------------------------------

    def _start_grpc_server(self) -> None:
        """Start the gRPC server if not already running."""
        if self.__grpc_server is not None:
            return
        from .grpc_srv import serve

        self.__grpc_server, self.__grpc_loop = serve(
            self.__servicer,
            port=self.config.grpc.port,
            block=False,
        )

    def _stop_grpc_server(self) -> None:
        """Stop the gRPC server and its event loop."""
        if self.__grpc_server is not None and self.__grpc_loop is not None:
            import asyncio

            async def _shutdown() -> None:
                await self.__grpc_server.stop(grace=5)

            try:
                future = asyncio.run_coroutine_threadsafe(
                    _shutdown(), self.__grpc_loop
                )
                future.result(timeout=10)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping gRPC server: %s", exc)

            self.__grpc_loop.call_soon_threadsafe(self.__grpc_loop.stop)
            self.__grpc_server = None
            self.__grpc_loop = None

    def _cleanup_room(self) -> None:
        """Release any previously allocated room."""
        if self.__room and self.__allocator is not None:
            try:
                self.__allocator.release_room(self.__room)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to release room %s: %s", self.__room, exc)
            self.__room = None

    # ------------------------------------------------------------------
    # State / observation helpers
    # ------------------------------------------------------------------

    def _wait_for_all_states(self, timeout: float = 30.0) -> None:
        """Wait for every registered training agent to send a state (batch)."""
        states = self.__servicer.get_states(timeout=timeout)
        if states is None:
            raise TimeoutError("Timeout waiting for states from all agents")
        # Verify that every expected unum is present.
        missing = self.agent_team_unums - set(states.keys())
        if missing:
            raise TimeoutError(
                f"Timeout waiting for states from unums: {missing}"
            )

    def _collect_world_models(self) -> dict[int, pb2.WorldModel]:
        """Return the latest world model for each agent unum."""
        wms: dict[int, pb2.WorldModel] = {}
        for unum in self.agent_team_unums:
            state = self.__servicer.last_state(unum)
            if state is not None:
                wms[unum] = state.world_model
        return wms

    def _collect_observations(self) -> dict[str, np.ndarray]:
        """Build observation vectors from world models for all training agents."""
        obs: dict[str, np.ndarray] = {}
        for agent_id in self._agent_id_list:
            _, unum = self._parse_agent_id(agent_id)
            state = self.__servicer.last_state(unum)
            if state is not None:
                obs[agent_id] = observation.extract(state.world_model)
            else:
                obs[agent_id] = np.zeros(self.obs_dim, dtype=np.float32)
        return obs


    @staticmethod
    def _action_to_proto(action: int) -> Any:
        """Convert a discrete action integer to a protobuf ``PlayerActions``."""
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


    def _calc_reward(
        self,
        agent_id: str,
        prev_wms: dict[int, pb2.WorldModel],
        curr_wms: dict[int, pb2.WorldModel],
    ) -> float:
        """Compute reward for *agent_id* from consecutive world models."""
        _, unum = self._parse_agent_id(agent_id)
        prev_wm = prev_wms.get(unum)
        curr_wm = curr_wms.get(unum)
        if prev_wm is None or curr_wm is None:
            return 0.0

        ret = reward.calculate(
            prev_obs=prev_wm,
            curr_obs=curr_wm,
            prev_truth=None,
            curr_truth=None,
        )
        return ret

    def _check_done(
        self, curr_wms: dict[int, pb2.WorldModel],
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Determine termination/truncation from world models."""

        # Use the first available world model.
        wm = next((w for w in curr_wms.values() if w is not None), None)

        game_over = wm is not None and wm.game_mode_type == pb2.TimeOver
        truncated = self._step_count >= self.room_schema.stopping.time_up

        terminateds: dict[str, bool] = {aid: game_over for aid in self._agent_id_list}
        terminateds["__all__"] = game_over

        truncateds: dict[str, bool] = {aid: truncated for aid in self._agent_id_list}
        truncateds["__all__"] = truncated

        return terminateds, truncateds

    def _collect_infos(
        self,
        curr_wms: dict[int, pb2.WorldModel],
    ) -> dict[str, dict[str, Any]]:
        """Build per-agent info dicts from world models."""
        infos: dict[str, dict[str, Any]] = {}
        for agent_id in self._agent_id_list:
            _, unum = self._parse_agent_id(agent_id)
            wm = curr_wms.get(unum)
            if wm is not None:
                infos[agent_id] = {
                    "scores": {
                        "our": wm.our_team_score,
                        "their": wm.their_team_score,
                    },
                    "step": self._step_count,
                    "cycle": wm.cycle,
                }
            else:
                infos[agent_id] = {"step": self._step_count}
        return infos

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Observation dimension (124)."""
        return observation.dim()

    @staticmethod
    def _parse_agent_id(agent_id: str) -> tuple[str, int]:
        """Parse ``"left_3"`` → ``("left", 3)``."""
        side, unum_str = agent_id.split("_", 1)
        return side, int(unum_str)
