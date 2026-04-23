"""RCSS multi-agent Gymnasium environment.

RCSSEnv is a Ray/RLlib-compatible ``MultiAgentEnv`` that requests simulation
rooms from a REST allocator and exchanges WorldModel / PlayerActions with
SoccerSimulationProxy sidecars via gRPC.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

from schema import GameServerSchema, TeamSide
from config import EnvConfig, BhvConfig

from . import obs as observation
from . import reward

from .action import Action
from .action_mask import ActionMaskResolver
from client.base.allocator import AllocatorClient
from client.room import RoomClient
from .grpc_srv.proto import pb2
from .grpc_srv.servicer import GameServicer

logger = logging.getLogger(__name__)


class RCSSEnv(MultiAgentEnv):
    """Multi-agent RCSS environment.

    Lifecycle:
      1. __init__: parse config, register agent unums, build obs/action spaces.
      2. reset:    release old room -> start gRPC server -> reset servicer -> allocate new room -> collect initial states.
      3. step:     encode & send actions -> collect new states -> compute obs/reward/terminated/truncated/info.
      4. close:    release room and stop gRPC server.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: EnvConfig) -> None:
        self.__cfg = config

        self.agent_team = self.schema.teams.agent_team
        self.agent_team_unums = set([agent.unum for agent in self.agent_team.ssp_agents()])

        # All agents share the same observation / action spaces
        _act_space = Action.space_schema()

        _obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        _obs_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            ActionMaskResolver.OBSERVATION_KEY: spaces.Box(
                low=0, high=1, shape=(Action.n_actions(),), dtype=np.int8
            ),
        })

        # Sorted list of agent ids (by unum)
        self.agents = list(sorted(self.agent_team_unums))

        self.observation_spaces = {unum: _obs_space for unum in self.agents}
        self.action_spaces = {unum: _act_space for unum in self.agents}

        super().__init__()

        # Mutable state
        self.__step_count: int = 0

        self.__loop = None

        # gRPC components (lazily initialized)
        self.__servicer: GameServicer | None = None
        self.__grpc_server: Any = None
        self.__grpc_loop: Any = None  # asyncio event loop for the gRPC aio server

        # Allocator client and current room
        self.__allocator: AllocatorClient | None = None
        self.__room: RoomClient | None = None

        # Per-agent State cache from the previous step, used for reward computation
        self.__prev_states: dict[int, pb2.State] = {}
        self.__curr_states: dict[int, pb2.State] = {}
        self.__player_by_unum = {
            player.unum: player
            for player in self.agent_team.players
        }

        self.__action_masks: dict[int, ActionMaskResolver] = {
            player.unum: ActionMaskResolver(player)
            for player in self.agent_team.players
        }

        self._setup()

    @property
    def config(self) -> EnvConfig:
        return self.__cfg

    @property
    def bhv(self) -> BhvConfig:
        return self.config.bhv

    @property
    def schema(self) -> GameServerSchema:
        """Return the RoomSchema used by this environment."""
        return self.config.room

    @property
    def allocator(self) -> AllocatorClient:
        if self.__allocator is None:
            raise RuntimeError("AllocatorClient not initialized")
        return self.__allocator

    @property
    def room(self) -> RoomClient:
        if self.__room is None:
            raise RuntimeError("RoomClient not initialized")
        return self.__room

    def has_room(self) -> bool:
        return self.__room is not None

    def has_allocator(self) -> bool:
        return self.__allocator is not None

    def _setup(self) -> None:
        """Initialize the allocator client and gRPC servicer, and register all agent unums."""
        self.__allocator = AllocatorClient(self.config.allocator)
        self.__servicer = GameServicer()

        for unum in self.agent_team_unums:
            self.__get_servicer().register(unum)

    def __get_servicer(self) -> GameServicer:
        if self.__servicer is None:
            raise RuntimeError("GameServicer not initialized")
        return self.__servicer

    @property
    def _agent_side(self) -> str:
        """Return the side string for the agent team ("left" / "right")."""
        return "left" if self.agent_team.side == TeamSide.LEFT else "right"

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, Any]]]:
        """Reset the environment: allocate a new simulation room and return initial observations."""
        super().reset(seed=seed, options=options)

        # Clear episode-local caches up-front so a failed reset cannot leak the
        # previous episode's state into the next one.
        self.__reset_episode_state()

        # 1. Release the previous room
        self._cleanup_room()

        # 2. Ensure the gRPC server is running
        self._start_grpc_server()

        # 3. Flush servicer internal buffers
        self.__reset_servicer_state()

        # 4. Request a new room from the allocator
        schema = self.config.room
        try:
            room = self.allocator.request_room(schema)
        except RuntimeError as exc:
            raise gymnasium.error.ResetNeeded(
                "Cannot allocate simulation room"
            ) from exc
        self.__room = room

        try:
            self.room.rcss.trainer.start()
            logger.info("Requested room %s from allocator and started simulation", self.room.info.name)
            # 5. Wait for all agent sidecars to send their initial states
            states = self.__collect_states()

        except Exception as e:
            self._cleanup_room()
            raise gymnasium.error.ResetNeeded(
                "Failed to start simulation or collect initial states, likely due to a communication issue with the simulation. "
            ) from e

        # 6. Build initial observations and info dicts
        self.__curr_states = states
        obs = self.__states_to_obs(states)
        infos: dict[int, dict[str, Any]] = {unum: {} for unum in self.agent_team_unums}
        return obs, infos

    def step(
        self,
        action_dict: dict[int, Any],
    ) -> tuple[
        dict[int, dict[str, np.ndarray]],
        dict[int, float],
        dict[Any, bool],
        dict[Any, bool],
        dict[int, dict[str, Any]],
    ]:
        """Execute one simulation step: send actions -> collect new states -> compute outputs."""
        self.__step_count += 1

        # 1. Encode each agent's action to a protobuf message and send them
        actions = self.__gather_actions(action_dict)
        self.__get_servicer().send_actions(actions)

        # 2. Collect new states and compare with the previous step for rewards
        self.__prev_states = self.__curr_states
        curr_states = self.__collect_states()
        self.__curr_states = curr_states

        rewards = {
            unum: self.__calc_reward(unum, self.__prev_states.get(unum), self.__curr_states.get(unum))
            for unum in self.agent_team_unums
            if unum in curr_states
        }

        curr_wms = {unum: state.world_model for unum, state in curr_states.items()}

        terminateds, truncateds = self.__check_done(curr_wms)

        obs = self.__states_to_obs(curr_states)
        infos = self.__collect_infos(curr_wms)

        return obs, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        """Release external resources: room + gRPC server."""
        self._cleanup_room()
        self._stop_grpc_server()

    # ------------------------------------------------------------------
    # Room / gRPC lifecycle helpers
    # ------------------------------------------------------------------
    def _start_grpc_server(self) -> None:
        """
        Start the gRPC server if it is not already running.
        MUST call BEFORE requesting the room allocation
        """
        if self.__grpc_server is not None:
            return
        from .grpc_srv.servicer import serve

        self.__grpc_server, actual_port, self.__grpc_loop = serve(
            self.__get_servicer(),
            port=self.config.grpc.port,
            block=False,
        )

        self.__sync_room_grpc_port(actual_port)

    def _stop_grpc_server(self) -> None:
        """Gracefully stop the gRPC server and its event loop."""
        if self.__grpc_server is not None and self.__grpc_loop is not None:
            import asyncio

            async def _shutdown() -> None:
                await self.__grpc_server.stop(grace=5)

            try:
                future = asyncio.run_coroutine_threadsafe(
                    _shutdown(), self.__grpc_loop
                )
                future.result(timeout=10)
            except Exception as exc:
                logger.warning("Error stopping gRPC server: %s", exc)

            self.__grpc_loop.call_soon_threadsafe(self.__grpc_loop.stop)
            self.__grpc_server = None
            self.__grpc_loop = None

    def _cleanup_room(self) -> None:
        """Release the currently allocated room via the allocator."""
        if self.has_room():
            try:
                self.room.release()
            except Exception as exc:
                logger.warning("Failed to release room %s: %s", self.room.info.name, exc)
            self.__room = None

    def __sync_room_grpc_port(self, port: int) -> None:
        """Mirror the bound gRPC port into every SSP agent policy in the room schema."""
        self.config.grpc.port = port
        for player in self.agent_team.ssp_agents():
            player.policy.grpc_port = port

    # ------------------------------------------------------------------
    # State / observation helpers
    # ------------------------------------------------------------------

    def __prev_state(self, unum: int) -> pb2.State | None:
        """Return the cached State from the previous step for the given unum."""
        return self.__prev_states.get(unum)

    def __prev_obs_wm(self, unum: int) -> pb2.WorldModel | None:
        """Return the observation WorldModel from the previous step for the given unum."""
        if (prev_state := self.__prev_state(unum)) is None: return None
        return prev_state.world_model

    def __latest_state(self, unum: int) -> pb2.State | None:
        return self.__curr_states.get(unum)

    def __latest_obs_wm(self, unum: int) -> pb2.WorldModel | None:
        if (latest_state := self.__latest_state(unum)) is None:
            return None
        return latest_state.world_model

    def __prev_truth_wm(self, unum: int) -> pb2.WorldModel | None:
        """Return the full-information WorldModel from the previous step for the given unum."""
        if (prev_state := self.__prev_state(unum)) is None: return None
        return prev_state.full_world_model

    def __collect_states(self, timeout_s: float = 180.0) -> dict[int, pb2.State]:
        """Block until all registered agents have sent their State."""
        try:
            states = self.__get_servicer().fetch_states(timeout=timeout_s)
            if len(missing := self.agent_team_unums.difference(states.keys())) != 0:
                raise ValueError(f"Missing states for unums: {sorted(missing)}")

            return states
        except Exception as exc:
            logger.warning("Timeout fetching states for world models: %s", exc)
            raise gymnasium.error.ResetNeeded(
                "Failed to fetch states for world models, likely due to a communication issue with the simulation. "
            ) from exc

    def __reset_episode_state(self) -> None:
        """Clear episode-local caches so reset failures do not leak prior state."""
        self.__step_count = 0
        self.__prev_states = {}
        self.__curr_states = {}

    def __reset_servicer_state(self) -> None:
        """Reset the servicer and restore the authoritative set of agent unums."""
        servicer = self.__get_servicer()
        servicer.reset()

        missing_unums = self.agent_team_unums.difference(servicer.unums)
        for unum in sorted(missing_unums):
            servicer.register(unum)

    def __states_to_obs(self, states: dict[int, pb2.State]) -> dict[int, dict[str, np.ndarray]]:
        """Convert states into masked observation payloads for each agent."""
        obs: dict[int, dict[str, np.ndarray]] = {}
        for unum, state in states.items():
            obs[unum] = {
                "obs": observation.extract(state.world_model).astype(np.float32),
                ActionMaskResolver.OBSERVATION_KEY: self.__action_mask(unum),
            }
        return obs

    def __validate_action_mask(self, unum: int, action_dict: dict[str, Any]) -> None:
        discrete_action = int(action_dict["actions"])
        act_mask = self.__action_mask(unum)
        if not Action.is_action_allowed(discrete_action, act_mask):
            action_name = Action.action_name(discrete_action)
            logger.warning(f"Action {action_name!r} (index={discrete_action}) is masked out for agent unum={unum}")

    def __gather_actions(self, action_dict: dict[int, Any]) -> dict[int, pb2.PlayerActions]:
        ret = {}
        for unum, act in action_dict.items():
            self.__validate_action_mask(unum, act)
            wm_opt = self.__latest_obs_wm(unum)

            body_act = Action.from_space(act).to_player_action()
            neck_act = self.bhv.neck.parse(wm_opt)
            view_act = self.bhv.view.parse(wm_opt)

            actions = (body_act, neck_act, view_act)
            actions = pb2.PlayerActions(actions=actions)

            ret[unum] = actions

        return ret

    def __action_mask(self, unum: int) -> np.ndarray:
        state = self.__latest_state(unum)
        wm = state.world_model if state is not None else None

        am = self.__action_masks.get(unum)
        ret = am.resolve(wm) if am is not None else Action.full_action_mask()

        return ret


    @staticmethod
    def __calc_reward(
        unum: int,
        prev_states: pb2.State | None,
        curr_states: pb2.State,
    ) -> float:
        """Compute the per-step reward for a single agent.

        The reward function receives both the observation WorldModel and the
        full-information WorldModel to support ground-truth-based reward shaping.
        """
        prev_obs = None
        prev_truth = None
        if prev_states is not None:
            prev_obs = prev_states.world_model
            prev_truth = prev_states.full_world_model

        if curr_states is None:
            logger.warning(f"Current state for unum={unum} is missing; cannot compute reward. ")
            return 0.0

        curr_obs = curr_states.world_model
        curr_truth = curr_states.full_world_model

        if prev_obs is None or prev_truth is None:
            return 0.0

        prev_obs_wm: pb2.WorldModel = prev_obs
        prev_truth_wm: pb2.WorldModel = prev_truth

        ret = reward.calculate(
            prev_obs_wm,
            prev_truth_wm,
            curr_obs,
            curr_truth,
        )
        return ret

    def __check_done(
        self, curr_wms: dict[int, pb2.WorldModel],
    ) -> tuple[dict[Any, bool], dict[Any, bool]]:
        """Determine terminated and truncated flags from the current WorldModels.

        terminated: game mode is TimeOver.
        truncated:  step count has reached StoppingEvents.time_up.
        """
        wm = next((w for w in curr_wms.values() if w is not None), None)

        game_over = wm is not None and wm.game_mode_type == pb2.TimeOver
        time_up = self.schema.stopping.time_up
        truncated = time_up is not None and self.__step_count >= time_up

        terminateds: dict[Any, bool] = {unum: game_over for unum in self.agent_team_unums}
        terminateds["__all__"] = game_over

        truncateds: dict[Any, bool] = {unum: truncated for unum in self.agent_team_unums}
        truncateds["__all__"] = truncated

        return terminateds, truncateds

    def __collect_infos(
        self,
        curr_wms: dict[int, pb2.WorldModel],
    ) -> dict[int, dict[str, Any]]:
        """Build the info dict for each agent, including scores, step count, and simulation cycle."""
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
    # Miscellaneous helpers
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Observation vector dimensionality."""
        return observation.dim()

    def __agent_name(self, unum: int) -> str:
        """Convert a unum to an agent ID string, e.g. ``"agent_left_7"``."""
        side = self._agent_side
        return f"agent_{side}_{unum}"
