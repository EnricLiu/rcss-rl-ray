"""Async gRPC Game servicer and server launcher.

GameServicer implements the ``Game`` gRPC service defined in service.proto.
It receives State messages from SoccerSimulationProxy sidecars, batches them
by timestep, and exchanges PlayerActions with the RL training loop.

The ``serve()`` helper launches the gRPC async server on a dedicated thread
so the caller (typically RCSSEnv) can interact with it from synchronous code.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from time import perf_counter
from typing import Any

import gymnasium
import grpc
from grpc import aio as grpc_aio

from .proto import pb2
from .proto import pb2_grpc
from .batch_queue import BatchQueue, BatchQueueError
from .truth_buffer import TruthWorldModelBuffer

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
STATE_GET_TIMEOUT_S = 5
ACTION_GET_TIMEOUT_S = 300
ACTION_SEND_TIMEOUT_S = 5

# Timeout for a single action queue put (prevents event-loop deadlock when queue is full)
ACTION_PUT_TIMEOUT_S = 5.0

SSP_RPC_SERVER_LANGUAGE = getattr(pb2, "PYThON", 0)


class ActionTimeoutError(RuntimeError):
    """Raised when a sidecar waits too long for RL actions."""


class SyncSetRuntimeError(RuntimeError):
    """Raised when a per-unum runtime failure would otherwise shrink the sync set."""


class GameServicer(pb2_grpc.GameServicer):
    """Async gRPC servicer bridging SoccerSimulationProxy sidecars and the RL loop.

    Each registered unum gets a pair of async queues: one for incoming States
    and one for outgoing PlayerActions.  A :class:`BatchQueue` synchronises
    states across all unums so that the RL loop receives a complete snapshot
    for each simulation cycle.
    """

    def __init__(self) -> None:

        self._batcher: BatchQueue[pb2.State] = BatchQueue()
        self._states_queues: dict[int, asyncio.Queue[tuple[int, pb2.State]]] = {}  # unum -> batched (ts, state)
        self._action_queues: dict[int, asyncio.Queue[pb2.PlayerActions]] = {}      # unum -> actions from RL loop

        self._states: dict[int, pb2.State] = {}  # latest fetched state per unum
        self._last_state_cycles: dict[int, int] = {}
        self._last_get_action_meta: dict[int, dict[str, Any]] = {}
        self._last_action_send_meta: dict[int, dict[str, Any]] = {}
        self._last_coach_truth_meta: dict[str, Any] = {}
        self._last_trainer_state_meta: dict[str, Any] = {}
        self._last_need_preprocess: dict[int, bool] = {}
        self._last_bye_meta: dict[int, dict[str, Any]] = {}
        self._last_planner_meta: dict[str, Any] = {}
        self._clients: dict[int, dict[str, Any]] = {}
        self._next_client_id = 1
        self._truth_buffer = TruthWorldModelBuffer(label="coach truth")
        self._trainer_buffer = TruthWorldModelBuffer(label="trainer")
        self._runtime_error: Exception | None = None

        self._server_params: pb2.ServerParam | None = None
        self._player_params: pb2.PlayerParam | None = None
        self._player_types: dict[int, pb2.PlayerType] = {}
        self._debug_mode: bool = False

        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind an asyncio event loop and perform an initial reset.

        Must be called before any synchronous bridge methods (fetch_states, send_actions, etc.).
        """
        self._loop = loop

    def register(self, unum: int) -> None:
        """Register a player unum, creating its state and action queues."""
        self._states_queues[unum] = asyncio.Queue(maxsize=1)
        self._action_queues[unum] = asyncio.Queue(maxsize=1)
        self._batcher.register(unum, self._states_queues[unum])

    def unregister(self, unum: int) -> None:
        """Unregister a player unum, removing its queues and cached state."""
        self._batcher.unregister(unum)
        self._states_queues.pop(unum, None)
        self._action_queues.pop(unum, None)
        self._states.pop(unum, None)

    @property
    def unums(self) -> frozenset[int]:
        """Return the set of currently registered unums."""
        return self._batcher.unums()

    def debug_snapshot(self) -> dict[str, Any]:
        """Return a structured runtime snapshot useful for timeout diagnostics."""
        return {
            "registered_unums": sorted(self._batcher.unums()),
            "cached_state_cycles": {
                str(unum): state.world_model.cycle
                for unum, state in sorted(self._states.items())
                if state.world_model is not None
            },
            "last_state_cycles": {
                str(unum): cycle
                for unum, cycle in sorted(self._last_state_cycles.items())
            },
            "last_get_action": {
                str(unum): dict(meta)
                for unum, meta in sorted(self._last_get_action_meta.items())
            },
            "last_action_send": {
                str(unum): dict(meta)
                for unum, meta in sorted(self._last_action_send_meta.items())
            },
            "last_coach_truth": dict(self._last_coach_truth_meta),
            "last_need_preprocess": {
                str(unum): value
                for unum, value in sorted(self._last_need_preprocess.items())
            },
            "last_bye": {
                str(client_id): dict(meta)
                for client_id, meta in sorted(self._last_bye_meta.items())
            },
            "last_planner": dict(self._last_planner_meta),
            "clients": {
                str(client_id): dict(meta)
                for client_id, meta in sorted(self._clients.items())
            },
            "truth_buffer": self._truth_buffer.snapshot(),
            "trainer_buffer": self._trainer_buffer.snapshot(),
            "last_trainer_state": dict(self._last_trainer_state_meta),
            "runtime_error": None if self._runtime_error is None else {
                "type": type(self._runtime_error).__name__,
                "message": str(self._runtime_error),
            },
            "state_queue_sizes": {
                str(unum): queue.qsize()
                for unum, queue in sorted(self._states_queues.items())
            },
            "action_queue_sizes": {
                str(unum): queue.qsize()
                for unum, queue in sorted(self._action_queues.items())
            },
            "batcher": self._batcher.snapshot(),
        }

    def _set_runtime_error(self, exc: Exception) -> None:
        if self._runtime_error is None:
            self._runtime_error = exc

    def _raise_if_runtime_error(self) -> None:
        if self._runtime_error is None:
            return
        raise gymnasium.error.ResetNeeded(str(self._runtime_error)) from self._runtime_error

    # ------------------------------------------------------------------
    # gRPC RPC handlers (called by the sidecar)
    # ------------------------------------------------------------------

    async def _abort_or_raise(
        self,
        context: grpc.aio.ServicerContext | None,
        code: grpc.StatusCode,
        message: str,
    ) -> None:
        if context is not None:
            await context.abort(code, message)
        raise ValueError(message)

    @staticmethod
    def _has_field(message: Any, field: str) -> bool:
        try:
            return bool(message.HasField(field))
        except (AttributeError, ValueError):
            return False

    async def Register(
        self, request: pb2.RegisterRequest, context: grpc.aio.ServicerContext | None
    ) -> pb2.RegisterResponse:
        """Register an SSP v2 client and return its server-side client id."""
        uniform_number = int(request.uniform_number)
        agent_type = int(request.agent_type)

        if request.agent_type == pb2.PlayerT and uniform_number not in self._batcher.unums():
            await self._abort_or_raise(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Register received unconfigured player uniform_number={uniform_number}",
            )

        client_id = self._next_client_id
        self._next_client_id += 1
        response = pb2.RegisterResponse(
            client_id=client_id,
            agent_type=request.agent_type,
            team_name=request.team_name,
            uniform_number=uniform_number,
            rpc_server_language_type=SSP_RPC_SERVER_LANGUAGE,
        )
        self._clients[client_id] = {
            "agent_type": agent_type,
            "team_name": request.team_name,
            "uniform_number": uniform_number,
            "status": "connected",
            "rpc_version": request.rpc_version,
            "connected_monotonic_s": round(perf_counter(), 6),
        }
        logger.debug(
            "Registered SSP v2 client_id=%d agent_type=%s team=%s unum=%d",
            client_id,
            agent_type,
            request.team_name,
            uniform_number,
        )
        return response

    async def SendByeCommand(
        self, request: pb2.RegisterResponse, context: grpc.aio.ServicerContext | None
    ) -> pb2.Empty:
        """Record SSP v2 client shutdown notifications."""
        client_id = int(request.client_id)
        meta = {
            "agent_type": int(request.agent_type),
            "team_name": request.team_name,
            "uniform_number": int(request.uniform_number),
            "received_monotonic_s": round(perf_counter(), 6),
        }
        self._last_bye_meta[client_id] = meta
        if client_id in self._clients:
            self._clients[client_id].update({
                "status": "disconnected",
                "disconnected_monotonic_s": meta["received_monotonic_s"],
            })
        logger.debug("Received SSP v2 bye command: client_id=%d meta=%s", client_id, meta)
        return pb2.Empty()

    async def GetBestPlannerAction(
        self,
        request: pb2.BestPlannerActionRequest,
        context: grpc.aio.ServicerContext | None,
    ) -> pb2.BestPlannerActionResponse:
        """Return a deterministic fallback planner choice without changing RL action space."""
        indices = sorted(int(index) for index in request.pairs.keys())
        selected_index = indices[0] if indices else 0
        self._last_planner_meta = {
            "client_id": int(request.register_response.client_id),
            "n_pairs": len(indices),
            "selected_index": selected_index,
            "received_monotonic_s": round(perf_counter(), 6),
        }
        return pb2.BestPlannerActionResponse(index=selected_index)

    def _request_unum(self, request: pb2.State) -> int | None:
        register_unum: int | None = None
        if self._has_field(request, "register_response"):
            register_unum = int(request.register_response.uniform_number)

        wm_unum: int | None = None
        if self._has_field(request, "world_model") and self._has_field(request.world_model, "self"):
            wm_unum = int(request.world_model.self.uniform_number)

        if register_unum and wm_unum and register_unum != wm_unum:
            raise ValueError(
                f"State uniform_number mismatch: register_response={register_unum}, world_model.self={wm_unum}"
            )
        return register_unum or wm_unum

    async def __get_action(self, unum: int, state: pb2.State) -> pb2.PlayerActions | None:
        """Submit a state to the batcher and wait for the RL loop to provide actions."""
        action_q = self._action_queues.get(unum)
        cycle = state.world_model.cycle if state.world_model else -1
        started_at = perf_counter()
        previous_meta = self._last_get_action_meta.get(unum)
        self._last_state_cycles[unum] = cycle
        self._last_get_action_meta[unum] = {
            "cycle": cycle,
            "received_monotonic_s": round(started_at, 6),
            "action_queue_size_before_wait": action_q.qsize() if action_q is not None else None,
        }

        if action_q is None:
            exc = SyncSetRuntimeError(
                f"GetPlayerActions received state for unregistered unum={unum} cycle={cycle}"
            )
            self._set_runtime_error(exc)
            logger.error("%s snapshot=%s", exc, self.debug_snapshot())
            raise exc

        if not self._batcher.has(unum):
            exc = SyncSetRuntimeError(
                f"Received state for batcher-unregistered unum={unum} cycle={cycle}"
            )
            self._set_runtime_error(exc)
            # Preserve the sync set and fail fast instead of silently shrinking it.
            # self.unregister(unum)
            logger.error("%s snapshot=%s", exc, self.debug_snapshot())
            raise exc

        if previous_meta is not None:
            previous_cycle = previous_meta.get("cycle")
            logger.debug(
                "__get_action: unum=%d cycle=%d arrived after previous_cycle=%s (delta_cycle=%s)",
                unum,
                cycle,
                previous_cycle,
                (cycle - previous_cycle) if isinstance(previous_cycle, int) else None,
            )

        logger.debug("__get_action: unum=%d cycle=%d — submitting to batcher", unum, cycle)
        try:
            await self._batcher.put(unum, state.world_model.cycle, state)
        except BatchQueueError as exc:
            wrapped = SyncSetRuntimeError(
                f"Failed to submit state for unum={unum} cycle={cycle} to batcher: {exc}"
            )
            self._set_runtime_error(wrapped)
            logger.error("%s snapshot=%s", wrapped, self.debug_snapshot())
            raise wrapped from exc
        logger.debug("__get_action: unum=%d cycle=%d — waiting for action (timeout=%ds)", unum, cycle, ACTION_GET_TIMEOUT_S)

        try:
            actions = await asyncio.wait_for(
                action_q.get(), timeout=ACTION_GET_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            self._last_get_action_meta[unum]["timed_out_waiting_for_action"] = True
            self._last_get_action_meta[unum]["waited_s"] = round(perf_counter() - started_at, 6)
            exc = ActionTimeoutError(
                f"Timed out waiting for RL actions for unum={unum} cycle={cycle} after {ACTION_GET_TIMEOUT_S}s"
            )
            self._set_runtime_error(exc)
            logger.warning(
                "Timeout waiting for actions for unum=%d cycle=%d — "
                "batcher_unums=%s action_q_size=%d snapshot=%s; marking runtime as failed",
                unum, cycle,
                sorted(self._batcher.unums()),
                action_q.qsize(),
                self.debug_snapshot(),
            )
            raise exc

        self._last_get_action_meta[unum]["timed_out_waiting_for_action"] = False
        self._last_get_action_meta[unum]["waited_s"] = round(perf_counter() - started_at, 6)
        logger.debug("__get_action: unum=%d cycle=%d — action received", unum, cycle)
        return actions

    async def GetPlayerActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.PlayerActions:
        """Handle a GetPlayerActions RPC from the sidecar."""
        try:
            unum = self._request_unum(request)
        except ValueError as exc:
            self._set_runtime_error(exc)
            await self._abort_or_raise(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb2.PlayerActions()

        if unum is None:
            logger.warning(
                "GetPlayerActions received state with no register_response or self info; returning empty actions"
            )
            return pb2.PlayerActions()

        self._last_need_preprocess[unum] = bool(request.need_preprocess)
        actions = await self.__get_action(unum, request)
        return actions or pb2.PlayerActions()

    async def GetCoachActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.CoachActions:
        """Store the coach truth WorldModel and return empty coach actions."""
        if not request.HasField("world_model"):
            logger.warning("GetCoachActions received state with no world_model")
            return pb2.CoachActions()

        wm = request.world_model
        await self._truth_buffer.put(wm)
        self._last_coach_truth_meta = {
            "cycle": wm.cycle,
            "client_id": int(request.register_response.client_id) if self._has_field(request, "register_response") else None,
            "received_monotonic_s": round(perf_counter(), 6),
            "game_mode_type": wm.game_mode_type,
        }
        logger.debug("GetCoachActions stored coach truth world model (cycle=%d)", wm.cycle)
        return pb2.CoachActions()

    async def GetTrainerActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.TrainerActions:
        """Store the trainer global WorldModel and return empty trainer actions."""
        if not request.HasField("world_model"):
            logger.warning("GetTrainerActions received state with no world_model")
            return pb2.TrainerActions()

        wm = request.world_model
        await self._trainer_buffer.put(wm)
        self._last_trainer_state_meta = {
            "cycle": wm.cycle,
            "client_id": int(request.register_response.client_id) if self._has_field(request, "register_response") else None,
            "received_monotonic_s": round(perf_counter(), 6),
            "game_mode_type": wm.game_mode_type,
        }
        logger.debug("GetTrainerActions stored trainer world model (cycle=%d)", wm.cycle)
        return pb2.TrainerActions()

    async def SendInitMessage(
        self, request: pb2.InitMessage, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Handle a SendInitMessage RPC, storing the debug-mode flag."""
        self._debug_mode = request.debug_mode
        register_response = request.register_response if self._has_field(request, "register_response") else None
        logger.debug(
            "Received InitMessage: client_id=%s agent_type=%s debug_mode=%s",
            None if register_response is None else register_response.client_id,
            None if register_response is None else register_response.agent_type,
            request.debug_mode,
        )
        return pb2.Empty()

    async def SendServerParams(
        self, request: pb2.ServerParam, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Store the server parameters sent by the simulation."""
        self._server_params = request
        logger.debug("Received ServerParam")
        return pb2.Empty()

    async def SendPlayerParams(
        self, request: pb2.PlayerParam, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Store the player parameters sent by the simulation."""
        self._player_params = request
        logger.debug("Received PlayerParam")
        return pb2.Empty()

    async def SendPlayerType(
        self, request: pb2.PlayerType, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Store a player type definition sent by the simulation."""
        self._player_types[request.id] = request
        logger.debug("Received PlayerType id=%d", request.id)
        return pb2.Empty()

    # ------------------------------------------------------------------
    # Async internal helpers
    # ------------------------------------------------------------------

    async def __fetch_states(self, timeout: float) -> dict[int, pb2.State]:
        """Await states from all registered unums' output queues."""
        queue_items = list(self._states_queues.items())
        if not queue_items:
            logger.debug("__fetch_states: no state queues registered — returning cached states %s", sorted(self._states.keys()))
            return self._states.copy()

        self._batcher.raise_if_failed()

        expected_unums = sorted(u for u, _ in queue_items)
        logger.debug(
            "__fetch_states: waiting for states from unums=%s timeout=%.1fs snapshot=%s",
            expected_unums,
            timeout,
            self.debug_snapshot(),
        )

        tasks = [
            asyncio.wait_for(queue.get(), timeout=timeout)
            for _, queue in queue_items
        ]

        res = await asyncio.gather(*tasks, return_exceptions=True)
        failures: list[tuple[int, Exception]] = []
        for (unum, _), item in zip(queue_items, res):
            if isinstance(item, Exception):
                logger.error(
                    "__fetch_states: timeout/error waiting for state from unum=%d (%s); preserving sync set and failing. batcher_unums=%s snapshot=%s",
                    unum, type(item).__name__, sorted(self._batcher.unums()), self.debug_snapshot(),
                )
                # Preserve the sync set and fail fast instead of silently shrinking it.
                # self.unregister(unum)
                failures.append((unum, item))
                continue

            _, state = item
            cycle = state.world_model.cycle if state.world_model else -1
            logger.debug("__fetch_states: received state unum=%d cycle=%d", unum, cycle)
            self._states[unum] = state
            self._last_state_cycles[unum] = cycle

        if failures:
            failing_unum, failing_exc = failures[0]
            exc = SyncSetRuntimeError(
                f"Failed to fetch aligned state for unum={failing_unum}: {type(failing_exc).__name__}: {failing_exc}"
            )
            self._set_runtime_error(exc)
            raise exc from failing_exc

        logger.debug("__fetch_states: done — returning states for unums=%s", sorted(self._states.keys()))
        return self._states.copy()

    async def __send_action(
        self, unum: int, action: pb2.PlayerActions
    ) -> None:
        """Put an action into the given unum's action queue."""
        action_q = self._action_queues.get(unum)
        if action_q is None:
            raise RuntimeError(f"Player unum={unum} not registered")
        if action_q.full():
            logger.warning(
                "__send_action: unum=%d action_queue is FULL (size=%d maxsize=%d) — "
                "previous action was not consumed; this may block the event loop",
                unum, action_q.qsize(), action_q.maxsize,
            )
        try:
            await asyncio.wait_for(action_q.put(action), timeout=ACTION_PUT_TIMEOUT_S)
            self._last_action_send_meta[unum] = {
                "queued_monotonic_s": round(perf_counter(), 6),
                "queue_size_after_put": action_q.qsize(),
                "n_actions": len(action.actions),
                "last_known_state_cycle": self._last_state_cycles.get(unum),
            }
            logger.debug("__send_action: unum=%d action enqueued", unum)
        except asyncio.TimeoutError:
            logger.error(
                "__send_action: unum=%d timed out putting action (queue full after %.1fs); "
                "sidecar may have disconnected; snapshot=%s",
                unum, ACTION_PUT_TIMEOUT_S, self.debug_snapshot(),
            )
            raise

    async def __send_actions(
        self, actions: dict[int, pb2.PlayerActions]
    ) -> set[int]:
        """Send actions to all specified unums concurrently."""
        logger.debug("__send_actions: sending actions to unums=%s", sorted(actions.keys()))
        tasks = [
            self.__send_action(unum, action)
            for unum, action in actions.items()
        ]

        ret = set()
        res = await asyncio.gather(*tasks, return_exceptions=True)
        failures: list[tuple[int, Exception]] = []
        for unum, r in zip(actions.keys(), res):
            if isinstance(r, Exception):
                logger.error(
                    "__send_actions: error sending action for unum=%d; preserving sync set and failing: %s snapshot=%s",
                    unum,
                    r,
                    self.debug_snapshot(),
                )
                # Preserve the sync set and fail fast instead of silently shrinking it.
                # self.unregister(unum)
                failures.append((unum, r))
                continue

            ret.add(unum)

        if failures:
            failing_unum, failing_exc = failures[0]
            exc = SyncSetRuntimeError(
                f"Failed to send action for unum={failing_unum}: {type(failing_exc).__name__}: {failing_exc}"
            )
            self._set_runtime_error(exc)
            raise exc from failing_exc

        logger.debug("__send_actions: done, sent to unums=%s", sorted(ret))
        return ret

    async def __fetch_truth_world_model(
        self,
        cycle: int,
        timeout: float,
    ) -> pb2.WorldModel:
        """Await the exact-cycle coach truth WorldModel."""
        logger.debug(
            "__fetch_truth_world_model: waiting for cycle=%d timeout=%.1fs snapshot=%s",
            cycle,
            timeout,
            self._truth_buffer.snapshot(),
        )
        truth = await self._truth_buffer.get(cycle, timeout=timeout)
        logger.debug("__fetch_truth_world_model: received cycle=%d", truth.cycle)
        return truth

    async def __discard_truth_before(self, cycle: int) -> None:
        await self._truth_buffer.discard_before(cycle)

    async def __fetch_trainer_world_model(
        self,
        cycle: int,
        timeout: float,
    ) -> pb2.WorldModel:
        """Await the exact-cycle trainer global WorldModel."""
        logger.debug(
            "__fetch_trainer_world_model: waiting for cycle=%d timeout=%.1fs snapshot=%s",
            cycle,
            timeout,
            self._trainer_buffer.snapshot(),
        )
        state = await self._trainer_buffer.get(cycle, timeout=timeout)
        logger.debug("__fetch_trainer_world_model: received cycle=%d", state.cycle)
        return state

    async def __discard_trainer_before(self, cycle: int) -> None:
        await self._trainer_buffer.discard_before(cycle)

    # ------------------------------------------------------------------
    # Synchronous bridge (called from the RL thread)
    # ------------------------------------------------------------------

    def __run_coro(self, coro: Any, timeout: float | None = None) -> Any:
        """Schedule *coro* on the bound event loop and block until it completes."""
        try:
            self._raise_if_runtime_error()
        except Exception:
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            raise
        loop = self._loop
        if loop is None:
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                "GameServicer has no bound event loop.  "
                "Call serve() or bind_loop() first."
            )
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            result = future.result(timeout=timeout)
        except Exception as exc:
            if self._runtime_error is None:
                self._set_runtime_error(exc)
            self._raise_if_runtime_error()
            raise
        self._raise_if_runtime_error()
        return result

    def fetch_states(
        self, timeout: float = STATE_GET_TIMEOUT_S
    ) -> dict[int, pb2.State]:
        """Block until states from all registered unums are available and return them."""
        return self.__run_coro(
            self.__fetch_states(timeout=timeout),
            timeout=timeout+0.5,
        )

    def send_action(self, unum: int, action: pb2.PlayerActions) -> None:
        """Send an action to a single unum (blocking)."""
        self.__run_coro(self.__send_action(unum, action), timeout=ACTION_SEND_TIMEOUT_S)

    def send_actions(self, actions: dict[int, pb2.PlayerActions]) -> None:
        """Send actions to all specified unums (blocking)."""
        self.__run_coro(self.__send_actions(actions), timeout=ACTION_SEND_TIMEOUT_S)

    def fetch_truth_world_model(
        self,
        cycle: int,
        timeout: float = STATE_GET_TIMEOUT_S,
    ) -> pb2.WorldModel:
        """Block until the exact-cycle coach truth WorldModel is available."""
        return self.__run_coro(
            self.__fetch_truth_world_model(cycle=cycle, timeout=timeout),
            timeout=timeout + 0.5,
        )

    def discard_truth_before(self, cycle: int) -> None:
        """Drop buffered coach truth world models older than *cycle*."""
        self.__run_coro(self.__discard_truth_before(cycle), timeout=ACTION_SEND_TIMEOUT_S)

    def fetch_trainer_world_model(
        self,
        cycle: int,
        timeout: float = STATE_GET_TIMEOUT_S,
    ) -> pb2.WorldModel:
        """Block until the exact-cycle trainer global WorldModel is available."""
        return self.__run_coro(
            self.__fetch_trainer_world_model(cycle=cycle, timeout=timeout),
            timeout=timeout + 0.5,
        )

    def discard_trainer_before(self, cycle: int) -> None:
        """Drop buffered trainer world models older than *cycle*."""
        self.__run_coro(self.__discard_trainer_before(cycle), timeout=ACTION_SEND_TIMEOUT_S)

    def last_state(self, unum: int) -> pb2.State | None:
        """Return the last fetched State for a unum, or None."""
        return self._states.get(unum)

    def server_params(self) -> pb2.ServerParam | None:
        """Return the stored ServerParam, or None if not yet received."""
        return self._server_params

    def player_params(self) -> pb2.PlayerParam | None:
        """Return the stored PlayerParam, or None if not yet received."""
        return self._player_params

    def player_type(self, type_id: int) -> pb2.PlayerType | None:
        """Return the PlayerType with the given *type_id*, or None."""
        return self._player_types.get(type_id)

    def player_types(self) -> dict[int, pb2.PlayerType]:
        """Return a copy of all stored PlayerType definitions."""
        return dict(self._player_types)

    async def __reset_runtime(self) -> None:
        """Reset batching/runtime state while preserving registered unums."""
        unums = tuple(self._batcher.unums())
        logger.debug("__reset_runtime: resetting servicer, preserving unums=%s", sorted(unums))

        await self._batcher.reset()

        self._states.clear()
        self._last_state_cycles.clear()
        self._last_get_action_meta.clear()
        self._last_action_send_meta.clear()
        self._last_coach_truth_meta.clear()
        self._last_trainer_state_meta.clear()
        self._last_need_preprocess.clear()
        self._last_bye_meta.clear()
        self._last_planner_meta.clear()
        self._clients.clear()
        self._next_client_id = 1
        await self._truth_buffer.reset()
        await self._trainer_buffer.reset()
        self._runtime_error = None
        self._states_queues.clear()
        self._action_queues.clear()

        for unum in unums:
            self.register(unum)

        self._server_params = None
        self._player_params = None
        self._player_types = {}
        self._debug_mode = False

        self._batcher.run()
        logger.debug("__reset_runtime: complete, registered_unums=%s", sorted(self._batcher.unums()))

    def reset(self) -> None:
        """Clear all internal state and re-register previously known unums."""
        logger.debug("GameServicer.reset: called, loop_bound=%s", self._loop is not None)
        if self._loop is None:
            unums = tuple(self._batcher.unums())
            self._batcher = BatchQueue()
            self._states.clear()
            self._last_state_cycles.clear()
            self._last_get_action_meta.clear()
            self._last_action_send_meta.clear()
            self._last_coach_truth_meta.clear()
            self._last_trainer_state_meta.clear()
            self._last_need_preprocess.clear()
            self._last_bye_meta.clear()
            self._last_planner_meta.clear()
            self._clients.clear()
            self._next_client_id = 1
            self._truth_buffer = TruthWorldModelBuffer(label="coach truth")
            self._trainer_buffer = TruthWorldModelBuffer(label="trainer")
            self._runtime_error = None
            self._states_queues.clear()
            self._action_queues.clear()

            for unum in unums:
                self.register(unum)

            self._server_params = None
            self._player_params = None
            self._player_types = {}
            self._debug_mode = False
            return

        self.__run_coro(self.__reset_runtime())


# ------------------------------------------------------------------
# Server launcher helpers
# ------------------------------------------------------------------

async def _start(servicer: GameServicer, port: int) -> tuple[grpc_aio.Server, int]:
    srv = grpc_aio.server()
    pb2_grpc.add_GameServicer_to_server(servicer, srv)
    port = srv.add_insecure_port(f"[::]:{port}")
    await srv.start()
    logger.info("gRPC Game server started on port %d", port)
    return srv, port

def _run_aio_server(
    servicer: GameServicer,
    port: int,
    started: threading.Event,
    server_holder: list[tuple[grpc_aio.Server, int]],
    block: bool,
) -> None:
    """Thread target: create an asyncio event loop, start the gRPC server, and run."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server, port = loop.run_until_complete(_start(servicer, port))

    servicer.bind_loop(loop)

    server_holder.append((server, port))
    started.set()

    if block:
        loop.run_until_complete(server.wait_for_termination())
    else:
        # Keep the loop alive so async callbacks continue to work
        loop.run_forever()


def serve(
    servicer: GameServicer | None = None,
    port: int = 50051,
    block: bool = True,
) -> tuple[grpc_aio.Server, int, asyncio.AbstractEventLoop]:
    """Launch the gRPC Game server on a daemon thread.

    Args:
        servicer: An existing GameServicer instance, or None to create a new one.
        port: TCP port to listen on.
        block: If True the calling thread blocks until the server terminates;
               if False the server runs in the background.

    Returns:
        A ``(server, port, loop)`` tuple for the running gRPC async server and its event loop.
    """
    if servicer is None:
        servicer = GameServicer()

    started = threading.Event()
    server_holder: list[grpc_aio.Server] = []

    thread = threading.Thread(
        target=_run_aio_server,
        args=(servicer, port, started, server_holder, block),
        daemon=True,
        name=f"grpc-aio-{port}",
    )
    thread.start()

    started.wait(timeout=30.0)
    if not server_holder:
        raise RuntimeError("gRPC aio server failed to start")

    server, port = server_holder[0]
    loop = servicer._loop
    assert loop is not None, "Loop should have been bound by _run_aio_server"

    if block:
        thread.join()

    return server, port, loop
