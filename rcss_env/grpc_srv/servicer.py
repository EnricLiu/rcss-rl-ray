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
from typing import Any

import grpc
from grpc import aio as grpc_aio

from .proto import pb2
from .proto import pb2_grpc
from .batch_queue import BatchQueue

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
STATE_GET_TIMEOUT_S = 2
STATE_SEND_TIMEOUT_S = 2
ACTION_GET_TIMEOUT_S = 30
ACTION_SEND_TIMEOUT_S = 2


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

        self.reset()

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

    # ------------------------------------------------------------------
    # gRPC RPC handlers (called by the sidecar)
    # ------------------------------------------------------------------

    async def __get_action(self, unum: int, state: pb2.State) -> pb2.PlayerActions | None:
        """Submit a state to the batcher and wait for the RL loop to provide actions."""
        action_q = self._action_queues.get(unum)

        if action_q is None:
            logger.warning(
                "GetPlayerActions from unregistered unum=%d; returning empty actions",
                unum,
            )
            return None

        if not self._batcher.has(unum):
            self.unregister(unum)
            logger.error(
                "Received state for unregistered unum=%d; unregistering and returning empty actions",
                unum,
            )
            return None

        await self._batcher.put(unum, state.world_model.cycle, state)

        try:
            actions = await asyncio.wait_for(
                action_q.get(), timeout=ACTION_GET_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout waiting for actions for unum=%d; returning empty actions",
                unum,
            )
            return pb2.PlayerActions()

        return actions

    async def GetPlayerActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.PlayerActions:
        """Handle a GetPlayerActions RPC from the sidecar."""
        wm = request.world_model
        if wm is None or not wm.HasField("self"):
            logger.warning(
                "GetPlayerActions received state with no self info; returning empty actions"
            )
            return pb2.PlayerActions()

        actions = await self.__get_action(wm.self.id, request)
        return actions or pb2.PlayerActions()

    async def GetCoachActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.CoachActions:
        """Handle a GetCoachActions RPC (currently returns empty actions)."""
        logger.debug("GetCoachActions called (cycle=%d)", request.world_model.cycle)
        return pb2.CoachActions()

    async def GetTrainerActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.TrainerActions:
        """Handle a GetTrainerActions RPC (currently returns empty actions)."""
        logger.debug("GetTrainerActions called (cycle=%d)", request.world_model.cycle)
        return pb2.TrainerActions()

    async def SendInitMessage(
        self, request: pb2.InitMessage, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Handle a SendInitMessage RPC, storing the debug-mode flag."""
        self._debug_mode = request.debug_mode
        logger.info(
            "Received InitMessage: agent_type=%s, debug_mode=%s",
            request.agent_type,
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

    async def GetInitMessage(
        self, request: pb2.Empty, context: grpc.aio.ServicerContext
    ) -> pb2.InitMessageFromServer:
        """Handle a GetInitMessage RPC (currently returns an empty message)."""
        return pb2.InitMessageFromServer()

    # ------------------------------------------------------------------
    # Async internal helpers
    # ------------------------------------------------------------------

    async def __fetch_states(self, timeout: float) -> dict[int, pb2.State]:
        """Await states from all registered unums' output queues."""
        unums = self._states_queues.keys()
        tasks = [
            asyncio.wait_for(queue.get(), timeout=timeout)
            for queue in self._states_queues.values()
        ]

        res = await asyncio.gather(*tasks, return_exceptions=True)
        for unum, item in zip(unums, res):
            if isinstance(item, Exception):
                self.unregister(unum)
                logger.error(
                    "Timeout waiting for state for unum=%d; unregistered",
                    unum,
                )
                continue

            _, state = item
            self._states[unum] = state

        return self._states.copy()

    async def __send_action(
        self, unum: int, action: pb2.PlayerActions
    ) -> None:
        """Put an action into the given unum's action queue."""
        action_q = self._action_queues.get(unum)
        if action_q is None: raise RuntimeError(f"Player unum={unum} not registered")
        await action_q.put(action)

    async def __send_actions(
        self, actions: dict[int, pb2.PlayerActions]
    ) -> set[int]:
        """Send actions to all specified unums concurrently."""
        tasks = [
            self.__send_action(unum, action)
            for unum, action in actions.items()
        ]

        ret = set()
        res = await asyncio.gather(*tasks, return_exceptions=True)
        for unum, r in zip(actions.keys(), res):
            if isinstance(r, RuntimeError):
                logger.warning(f"Failed to send actions for unum={unum}: {r}")
            elif isinstance(r, Exception):
                self.unregister(unum)
                logger.error(f"Error sending actions for unum={unum}, unregistered: {r}")

            ret.add(unum)

        return ret

    # ------------------------------------------------------------------
    # Synchronous bridge (called from the RL thread)
    # ------------------------------------------------------------------

    def __run_coro(self, coro: Any, timeout: float | None = None) -> Any:
        """Schedule *coro* on the bound event loop and block until it completes."""
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                "GameServicer has no bound event loop.  "
                "Call serve() or bind_loop() first."
            )
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

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

    def reset(self) -> None:
        """Clear all internal state and re-register previously known unums."""
        unums = self._batcher.unums()
        self._states.clear()
        self._states_queues.clear()
        self._action_queues.clear()
        self._batcher.reset()

        for unum in unums:
            self.register(unum)

        self._server_params = None
        self._player_params = None
        self._player_types = {}
        self._debug_mode = False


# ------------------------------------------------------------------
# Server launcher helpers
# ------------------------------------------------------------------

def _run_aio_server(
    servicer: GameServicer,
    port: int,
    started: threading.Event,
    server_holder: list,
    block: bool,
) -> None:
    """Thread target: create an asyncio event loop, start the gRPC server, and run."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _start() -> grpc_aio.Server:
        srv = grpc_aio.server()
        pb2_grpc.add_GameServicer_to_server(servicer, srv)
        srv.add_insecure_port(f"[::]:{port}")
        await srv.start()
        logger.info("gRPC Game server started on port %d", port)
        return srv

    server = loop.run_until_complete(_start())

    servicer.bind_loop(loop)

    server_holder.append(server)
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
) -> tuple[grpc_aio.Server, asyncio.AbstractEventLoop]:
    """Launch the gRPC Game server on a daemon thread.

    Args:
        servicer: An existing GameServicer instance, or None to create a new one.
        port: TCP port to listen on.
        block: If True the calling thread blocks until the server terminates;
               if False the server runs in the background.

    Returns:
        A ``(server, loop)`` tuple for the running gRPC async server and its event loop.
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

    server = server_holder[0]
    loop = servicer._loop
    assert loop is not None, "Loop should have been bound by _run_aio_server"

    if block:
        thread.join()

    return server, loop
