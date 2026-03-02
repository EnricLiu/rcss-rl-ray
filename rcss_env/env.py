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
from ray.rllib.env import MultiAgentEnv

from schema import RoomSchema, PlayerSchema, PlayerInitState, TeamSide
from config import EnvConfig

import obs as observation
import reward

from .action import Action
from .allocator import AllocatorClient
from .grpc_srv import GameServicer, pb2

logger = logging.getLogger(__name__)


class RCSSEnv(MultiAgentEnv):
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
        self.__cfg = config

        self.agent_team = self.room_schema.teams.agent_team
        self.agent_team_unums = set([agent.unum for agent in self.agent_team.ssp_agents()])

        # Per-agent spaces (shared by all agents).
        _obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        _obs_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "act_mask": spaces.MultiBinary(Action.n_actions()),
        })

        _act_space = Action.space_schema()

        # Build agent ID set and per-agent space mappings for MultiAgentEnv.
        self.agents = list(sorted(self.agent_team_unums))

        self.observation_spaces = {unum: _obs_space for unum in self.agents}
        self.action_spaces = {unum: _act_space for unum in self.agents}

        super().__init__()

        # Mutable state (populated by :meth:`reset`).
        self.__step_count: int = 0

        # Remote-mode components (lazily initialised).
        self.__loop = None

        self.__servicer: GameServicer | None = None
        self.__grpc_server: Any = None
        self.__grpc_loop: Any = None  # asyncio event loop for gRPC aio server

        self.__allocator: Any = None
        self.__room: str | None = None

        self.__prev_states: dict[int, pb2.State] = {}

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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        """Reset the environment and return initial observations."""
        super().reset(seed=seed, options=options)

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
        states = self.__collect_states()

        # 6. Collect initial observations and infos.
        self.__step_count = 0
        self.__prev_states = {}
        obs = self.__states_to_obs(states)
        infos: dict[int, dict[str, Any]] = {unum: {} for unum in self.agent_team_unums}
        return obs, infos

    def step(
        self,
        action_dict: dict[int, Any],
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict[str, Any]],
    ]:
        """Execute one simulation step: send actions, wait for next states."""
        self.__step_count += 1

        # 1. Build proto actions for every agent and send as a batch.
        actions = {
            unum: Action.from_space(action).get_action()
            for unum, action in action_dict.items()
        }
        self.__servicer.send_actions(actions)

        # 2. Collect observations, rewards, termination, truncation.
        prev_states = self.__prev_states
        curr_states = self.__collect_states()
        self.__prev_states = curr_states

        obs = self.__states_to_obs(curr_states)
        rewards = {
            unum: self.__calc_reward(unum, prev_states.get(unum), curr_states.get(unum))
            for unum in self.agent_team_unums
        }

        curr_wms = {unum: state.world_model for unum, state in curr_states.items()}
        terminateds, truncateds = self.__check_done(curr_wms)
        infos = self.__collect_infos(curr_wms)

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

    def __prev_state(self, unum: int) -> pb2.State | None:
        """Return the last received state for a given agent unum."""
        return self.__prev_states.get(unum)

    def __prev_obs_wm(self, unum: int) -> pb2.WorldModel | None:
        """Return the world model from the last received state for a given agent unum."""
        if (prev_state := self.__prev_state(unum)) is None: return None
        return prev_state.world_model

    def __prev_truth_wm(self, unum: int) -> pb2.WorldModel | None:
        """Return the world model from the last received state for a given agent unum, with truth info."""
        if (prev_state := self.__prev_state(unum)) is None: return None
        return prev_state.full_world_model


    def __collect_states(self, timeout_s: float = 30.0) -> dict[int, pb2.State]:
        """Return the latest world model for each agent unum."""
        try:
            states = self.__servicer.fetch_states(timeout=timeout_s)
            return states
        except Exception as exc:
            logger.warning("Timeout fetching states for world models: %s", exc)
            raise gymnasium.error.ResetNeeded(
                "Failed to fetch states for world models, likely due to a communication issue with the simulation. "
            )

    @staticmethod
    def __states_to_obs(states: dict[int, pb2.State]) -> dict[int, np.ndarray]:
        obs = {
            unum: observation.extract(s.world_model).astype(np.float32)
            for unum, s in states.items()
        }
        return obs

    @staticmethod
    def __calc_reward(
        unum: int,
        prev_states: pb2.State | None,
        curr_states: pb2.State,
    ) -> float:
        """Compute reward for *agent_id* from consecutive world models."""

        prev_obs = None
        prev_truth = None
        if prev_states is not None:
            prev_obs = prev_states.world_model
            prev_truth = prev_states.full_world_model

        if curr_states is None:
            logger.warning(f"Current state for unum={unum} is missing; cannot compute reward. ")
            return 0.0  # No current state → zero reward (could also raise an error)

        curr_obs = curr_states.world_model
        curr_truth = curr_states.full_world_model

        ret = reward.calculate(
            prev_obs,
            curr_obs,
            prev_truth,
            curr_truth,
        )
        return ret


    def __check_done(
        self, curr_wms: dict[int, pb2.WorldModel],
    ) -> tuple[dict[int, bool], dict[int, bool]]:
        """Determine termination/truncation from world models."""

        # Use the first available world model.
        wm = next((w for w in curr_wms.values() if w is not None), None)

        game_over = wm is not None and wm.game_mode_type == pb2.TimeOver
        truncated = self.__step_count >= self.room_schema.stopping.time_up

        terminateds: dict[int, bool] = {unum: game_over for unum in self.agent_team_unums}
        terminateds["__all__"] = game_over

        truncateds: dict[int, bool] = {unum: truncated for unum in self.agent_team_unums}
        truncateds["__all__"] = truncated

        return terminateds, truncateds

    def __collect_infos(
        self,
        curr_wms: dict[int, pb2.WorldModel],
    ) -> dict[int, dict[str, Any]]:
        """Build per-agent info dicts from world models."""
        infos: dict[int, dict[str, Any]] = {}
        for unum in self.agent_team_unums:
            wm = curr_wms.get(unum)
            if wm is not None:
                infos[unum] = {
                    "scores": {
                        "our": wm.our_team_score,
                        "their": wm.their_team_score,
                    },
                    "step": self.__step_count,
                    "cycle": wm.cycle,
                }
            else:
                infos[unum] = {"step": self.__step_count}
        return infos

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Observation dimension (124)."""
        return observation.dim()

    def __agent_name(self, unum: int) -> str:
        """Convert a unum to an agent ID string, e.g. 7 → "agent_left_7"."""
        side = self._agent_side
        return f"agent_{side}_{unum}"

