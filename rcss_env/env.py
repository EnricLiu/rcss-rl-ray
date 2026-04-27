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
from pydantic import IPvAnyAddress
from gymnasium import spaces
from ray.util import get_node_ip_address
from ray.rllib.env import MultiAgentEnv

from schema import GameServerSchema, TeamSide, TeamSchema
from client.room import RoomClient
from client.base.allocator import AllocatorClient
from train.curriculum import CurriculumMixin

from .bhv import NeckViewBhv
from .reward import RewardFnMixin
from .config import EnvConfig
from .action import Action
from .action_mask import ActionMaskResolver
from . import obs as observation

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

        self.__schema = self.gen_schema()
        self.__reward_fn = self.config.curriculum.reward_fn()

        # All agents share the same observation / action spaces
        _act_space = Action.space_schema()
        _obs_space = spaces.Dict({
            "obs": spaces.Box(
                low=np.full((self.obs_dim,), -np.inf, dtype=np.float32),
                high=np.full((self.obs_dim,), np.inf, dtype=np.float32),
                dtype=np.float32,
            ),
            ActionMaskResolver.OBSERVATION_KEY: spaces.Box(
                low=np.zeros((Action.n_actions(),), dtype=np.int8),
                high=np.ones((Action.n_actions(),), dtype=np.int8),
                dtype=np.int8,
            ),
        })

        # Sorted list of agent ids (by unum)
        self.agents = list(sorted(self.agent_team_unums))

        self.observation_spaces = {unum: _obs_space for unum in self.agents}
        self.action_spaces = {unum: _act_space for unum in self.agents}

        super().__init__()

        # Mutable state
        self.__timestep: int = 0

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
        self.__prev_truth_world_model: pb2.WorldModel | None = None
        self.__curr_truth_world_model: pb2.WorldModel | None = None
        self.__last_reward_breakdowns: dict[int, dict[str, float]] = {}
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
    def agent_team(self) -> TeamSchema:
        return self.schema.teams.agent_team

    @property
    def agent_team_unums(self) -> set[int]:
        return set([agent.unum for agent in self.agent_team.ssp_agents()])

    @property
    def bhv(self) -> NeckViewBhv:
        return self.config.bhv

    @property
    def reward(self) -> RewardFnMixin:
        return self.__reward_fn

    @property
    def curriculum(self) -> CurriculumMixin:
        return self.config.curriculum

    @property
    def schema(self) -> GameServerSchema:
        """Return the RoomSchema used by this environment."""
        return self.__schema

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

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Return a structured snapshot of env/runtime state for timeout debugging."""
        return {
            "step_count": self.__timestep,
            "agent_unums": sorted(self.agent_team_unums),
            "prev_state_cycles": self.__state_cycles(self.__prev_states),
            "curr_state_cycles": self.__state_cycles(self.__curr_states),
            "prev_truth_cycle": self.__truth_cycle(self.__prev_truth_world_model),
            "curr_truth_cycle": self.__truth_cycle(self.__curr_truth_world_model),
            "has_room": self.has_room(),
            "room": {
                "name": self.room.info.name,
                "base_url_rcss": self.room.info.base_url_rcss,
                "base_url_mc": self.room.info.base_url_mc,
            } if self.has_room() else None,
            "servicer": self.__get_servicer().debug_snapshot(),
        }

    def gen_schema(self) -> GameServerSchema:
        self.__schema = self.curriculum.make_schema()
        self.__inject_agent_grpc_config()
        return self.schema

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

        # 0. make rand schema, set the reward fn
        self.__schema = self.gen_schema()
        self.__reward_fn = self.curriculum.reward_fn()

        # Clear episode-local caches up-front so a failed reset cannot leak the
        # previous episode's state into the next one.
        self.__reset_episode_state()

        # 1. Release the previous room
        self._cleanup_room()

        # 2. Ensure the gRPC server is running, sync the actual grpc server
        # addr to the self.__schema
        self._start_grpc_server()

        # 3. Flush servicer internal buffers
        self.__reset_servicer_state()

        try:
            self.__room = self.allocator.request_room(self.__schema)
        except RuntimeError as exc:
            raise gymnasium.error.ResetNeeded(
                "Cannot allocate simulation room"
            ) from exc

        try:
            self.room.rcss.trainer.start()
            logger.debug("Requested room %s from allocator and started simulation", self.room.info.name)
            # 5. Wait for all agent sidecars to send their initial states
            states = self.__collect_states()
            truth = self.__collect_truth_world_model(self.__aligned_state_cycle(states))
            logger.debug(
                "Reset: collected initial states for unums=%s cycles=%s truth_cycle=%d",
                sorted(states.keys()),
                self.__state_cycles(states),
                truth.cycle,
            )

        except Exception as e:
            self._cleanup_room()
            raise gymnasium.error.ResetNeeded(
                "Failed to start simulation or collect initial states, likely due to a communication issue with the simulation. "
            ) from e

        # 6. Build initial observations and info dicts
        self.__curr_states = states
        self.__curr_truth_world_model = truth
        obs = self.__states_to_obs(states)
        infos: dict[int, dict[str, Any]] = {unum: {} for unum in self.agent_team_unums}
        return obs, infos

    def __step(
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

        # 1. Encode each agent's action to a protobuf message and send them
        logger.debug(
            "Step %d: gathering actions for unums=%s from current_cycles=%s",
            self.__timestep,
            sorted(action_dict.keys()),
            self.__timestep,
        )
        actions = self.__gather_actions(action_dict)
        self.__get_servicer().send_actions(actions)

        # 2. Collect new states and compare with the previous step for rewards
        self.__prev_states = self.__curr_states
        self.__prev_truth_world_model = self.__curr_truth_world_model
        logger.debug("Step %d: waiting for states from simulation", self.__timestep)
        curr_states = self.__collect_states()
        self.__curr_states = curr_states

        curr_cycle = self.__aligned_state_cycle(curr_states)
        curr_truth = self.__collect_truth_world_model(curr_cycle)
        self.__curr_truth_world_model = curr_truth

        self.__timestep = curr_cycle

        self.__last_reward_breakdowns = {}
        rewards: dict[int, float] = {}
        for unum, curr_state in curr_states.items():
            if unum not in self.agent_team_unums:
                continue
            rewards[unum] = self.__calc_reward(
                unum,
                self.__prev_states.get(unum),
                curr_state,
                self.__prev_truth_world_model,
                curr_truth,
            )

        self.__get_servicer().discard_truth_before(curr_cycle)

        curr_wms = {unum: state.world_model for unum, state in curr_states.items()}

        terminateds, truncateds = self.__check_done(curr_wms)

        obs = self.__states_to_obs(curr_states)
        infos = self.__collect_infos(curr_wms)

        return obs, rewards, terminateds, truncateds, infos

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
        try:
            return self.__step(action_dict)
        except gymnasium.error.ResetNeeded as exc:
            logger.warning(
                "Reset needed during step: %s. Returning truncated terminal payload. Runtime diagnostics: %s",
                exc,
                self.runtime_diagnostics(),
            )
            return self.__reset_needed_step_result(exc)


    def close(self) -> None:
        """Release external resources: room + gRPC server."""
        self._cleanup_room()
        self._stop_grpc_server()

    @property
    def timestep(self) -> int:
        return self.__timestep

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

        actual_host = get_node_ip_address()
        self.__sync_room_grpc_server(actual_host, actual_port)

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

    def __sync_room_grpc_server(self, host: str, port: int) -> None:
        """Mirror the bound gRPC port into every SSP agent policy in the room schema."""
        _host = IPvAnyAddress(host)
        self.config.grpc.host = _host
        self.config.grpc.port = port

        self.__inject_agent_grpc_config()

    def __inject_agent_grpc_config(self):
        for player in self.agent_team.ssp_agents():
            player.policy.grpc_host = self.config.grpc.host
            player.policy.grpc_port = self.config.grpc.port
        if (coach := self.agent_team.coach) is not None:
            coach.policy.grpc_host = self.config.grpc.host
            coach.policy.grpc_port = self.config.grpc.port


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
        """Return the previous coach truth WorldModel."""
        return self.__prev_truth_world_model

    def __collect_states(self, timeout_s: float = 180.0) -> dict[int, pb2.State]:
        """
        Block until all registered agents have sent their State.
        Returns when States are aligned and GameModeType is PlayOn.
        """
        try:
            while True:
                states = self.__get_servicer().fetch_states(timeout=timeout_s)
                if len(missing := self.agent_team_unums.difference(states.keys())) != 0:
                    raise ValueError(f"Missing states for unums: {sorted(missing)}")

                keys = list(states.keys())
                if len(keys) == 0:
                    raise RuntimeError(f"No states returned by __collect_states!")

                if (wm := states[keys[0]].world_model).game_mode_type != pb2.GameModeType.PlayOn:
                    logger.debug(
                        "[__collect_states@ts=%d] GameMode=%s. Sending Idle Actions.",
                        wm.cycle,
                        wm.game_mode_type,
                    )
                    self.__get_servicer().send_actions(self.__idle_action(unums=set(states.keys())))
                else:
                    return states

        except Exception as exc:
            logger.warning(
                "Timeout fetching states for world models: %s runtime=%s",
                exc,
                self.runtime_diagnostics(),
            )
            raise gymnasium.error.ResetNeeded(
                "Failed to fetch states for world models, likely due to a communication issue with the simulation. "
            ) from exc

    def __collect_truth_world_model(
        self,
        cycle: int,
        timeout_s: float = 180.0,
    ) -> pb2.WorldModel:
        """Fetch the exact-cycle coach truth WorldModel used for reward computation."""
        try:
            truth = self.__get_servicer().fetch_truth_world_model(cycle, timeout=timeout_s)
            if truth.cycle != cycle:
                raise ValueError(
                    f"Fetched coach truth cycle={truth.cycle}, expected cycle={cycle}"
                )
            return truth
        except Exception as exc:
            logger.warning(
                "Timeout fetching coach truth world model for cycle=%d: %s runtime=%s",
                cycle,
                exc,
                self.runtime_diagnostics(),
            )
            raise gymnasium.error.ResetNeeded(
                f"Failed to fetch coach truth world model for cycle={cycle}. "
            ) from exc

    def __reset_episode_state(self) -> None:
        """Clear episode-local caches so reset failures do not leak prior state."""
        self.__timestep = 0
        self.__prev_states = {}
        self.__curr_states = {}
        self.__prev_truth_world_model = None
        self.__curr_truth_world_model = None
        self.__last_reward_breakdowns = {}

    def __reset_servicer_state(self) -> None:
        """Reset the servicer and restore the authoritative set of agent unums."""
        servicer = self.__get_servicer()
        servicer.reset()

        missing_unums = self.agent_team_unums.difference(servicer.unums)
        for unum in sorted(missing_unums):
            servicer.register(unum)

    def __coerce_obs_vector(self, unum: int, wm: pb2.WorldModel) -> np.ndarray:
        obs = np.asarray(observation.extract(wm), dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Observation shape mismatch for unum={unum}: expected={(self.obs_dim,)}, got={obs.shape}"
            )
        if not np.isfinite(obs).all():
            logger.warning(
                "Observation for unum=%d contains non-finite values; replacing them with finite float32 defaults",
                unum,
            )
            obs = np.nan_to_num(obs, copy=False)
        return obs

    def __coerce_action_mask(self, unum: int) -> np.ndarray:
        mask = np.asarray(self.__action_mask(unum))
        if mask.shape != (Action.n_actions(),):
            raise ValueError(
                f"Action mask shape mismatch for unum={unum}: expected={(Action.n_actions(),)}, got={mask.shape}"
            )
        if not np.issubdtype(mask.dtype, np.integer):
            mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        coerced = np.where(mask > 0, 1, 0).astype(np.int8, copy=False)
        return coerced

    def __states_to_obs(self, states: dict[int, pb2.State]) -> dict[int, dict[str, np.ndarray]]:
        """Convert states into masked observation payloads for each agent."""
        obs: dict[int, dict[str, np.ndarray]] = {}
        for unum, state in states.items():
            obs[unum] = {
                "obs": self.__coerce_obs_vector(unum, state.world_model),
                ActionMaskResolver.OBSERVATION_KEY: self.__coerce_action_mask(unum),
            }
        return obs

    def __zero_obs(self) -> dict[str, np.ndarray]:
        return {
            "obs": np.zeros((self.obs_dim,), dtype=np.float32),
            ActionMaskResolver.OBSERVATION_KEY: Action.full_action_mask().copy(),
        }

    def __terminal_obs(self) -> dict[int, dict[str, np.ndarray]]:
        cached_obs = self.__states_to_obs(self.__curr_states) if self.__curr_states else {}
        obs: dict[int, dict[str, np.ndarray]] = {}
        for unum in self.agent_team_unums:
            obs[unum] = cached_obs.get(unum, self.__zero_obs())
        return obs

    @staticmethod
    def __state_cycles(states: dict[int, pb2.State]) -> dict[int, int]:
        """Extract world-model cycles from a unum->State mapping."""
        ret: dict[int, int] = {}
        for unum, state in states.items():
            if state is not None and state.world_model is not None:
                ret[unum] = state.world_model.cycle
            else:
                logger.debug("State for unum=%d is missing or has no world_model; cannot extract cycle", unum)
        return ret

    @staticmethod
    def __truth_cycle(world_model: pb2.WorldModel | None) -> int | None:
        return None if world_model is None else int(world_model.cycle)

    def __aligned_state_cycle(self, states: dict[int, pb2.State]) -> int:
        cycles_by_unum = self.__state_cycles(states)
        if len(cycles_by_unum) != len(states):
            missing = sorted(set(states) - set(cycles_by_unum))
            raise ValueError(f"Missing world_model cycle for unums: {missing}")
        cycles = set(cycles_by_unum.values())
        if len(cycles) != 1:
            raise ValueError(f"Expected exactly one aligned state cycle, got {sorted(cycles)}")
        return next(iter(cycles))

    def __validate_action_mask(self, unum: int, action_dict: dict[str, Any]) -> None:
        discrete_action = int(action_dict["actions"])
        act_mask = self.__action_mask(unum)
        if not Action.is_action_allowed(discrete_action, act_mask):
            action_name = Action.action_name(discrete_action)
            logger.debug("Action %r (index=%d) is masked out for agent unum=%d", action_name, discrete_action, unum)

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

    def __idle_action(self, unums: set[int]) -> dict[int, pb2.PlayerActions]:
        ret = {}
        for unum in unums:
            wm_opt = self.__latest_obs_wm(unum)
            neck_act = self.bhv.neck.parse(wm_opt)
            view_act = self.bhv.view.parse(wm_opt)

            actions = (neck_act, view_act)
            actions = pb2.PlayerActions(actions=actions)

            ret[unum] = actions

        return ret

    def __action_mask(self, unum: int) -> np.ndarray:
        state = self.__latest_state(unum)
        wm = state.world_model if state is not None else None

        am = self.__action_masks.get(unum)
        ret = am.resolve(wm) if am is not None else Action.full_action_mask()

        return ret

    def __calc_reward(
        self,
        unum: int,
        prev_states: pb2.State | None,
        curr_states: pb2.State,
        prev_truth: pb2.WorldModel | None,
        curr_truth: pb2.WorldModel,
    ) -> float:
        """Compute the per-step reward for a single agent.

        The reward function receives player observations plus exact-cycle coach
        truth world models to support ground-truth-based reward shaping.
        """
        prev_obs = None
        if prev_states is not None:
            prev_obs = prev_states.world_model

        if curr_states is None:
            logger.warning("Current state for unum=%d is missing; cannot compute reward", unum)
            return 0.0

        curr_obs = curr_states.world_model

        if prev_obs is None or prev_truth is None:
            return 0.0

        prev_obs_wm: pb2.WorldModel = prev_obs
        prev_truth_wm: pb2.WorldModel = prev_truth

        ret = self.reward.compute(
            prev_obs_wm,
            prev_truth_wm,
            curr_obs,
            curr_truth,
        )

        self.__last_reward_breakdowns[unum] = self.reward.last_reward_breakdown

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
        truncated = time_up is not None and self.__timestep >= time_up

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
                    "step": self.__timestep,
                    "cycle": wm.cycle,
                }
            else:
                infos[unum] = {"step": self.__timestep}

            if reward_breakdown := self.__last_reward_breakdowns.get(unum):
                infos[unum]["reward_breakdown"] = reward_breakdown.copy()
        return infos

    def __reset_needed_infos(self, exc: gymnasium.error.ResetNeeded) -> dict[int, dict[str, Any]]:
        curr_wms = {
            unum: state.world_model
            for unum, state in self.__curr_states.items()
            if state is not None and state.world_model is not None
        }
        infos = self.__collect_infos(curr_wms)
        for unum in self.agent_team_unums:
            info = infos.setdefault(unum, {"step": self.__timestep})
            info["reset_needed"] = True
            info["error_type"] = type(exc).__name__
            info["error_message"] = str(exc)
        return infos

    def __reset_needed_step_result(
        self,
        exc: gymnasium.error.ResetNeeded,
    ) -> tuple[
        dict[int, dict[str, np.ndarray]],
        dict[int, float],
        dict[Any, bool],
        dict[Any, bool],
        dict[int, dict[str, Any]],
    ]:
        obs = self.__terminal_obs()
        rewards = {unum: 0.0 for unum in self.agent_team_unums}

        terminateds: dict[Any, bool] = {unum: False for unum in self.agent_team_unums}
        terminateds["__all__"] = False

        truncateds: dict[Any, bool] = {unum: True for unum in self.agent_team_unums}
        truncateds["__all__"] = True

        infos = self.__reset_needed_infos(exc)
        return obs, rewards, terminateds, truncateds, infos

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
