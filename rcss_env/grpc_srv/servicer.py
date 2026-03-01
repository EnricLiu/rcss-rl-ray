"""gRPC ``Game`` service implementation for SoccerSimulationProxy.

The SoccerSimulationProxy sidecar calls ``GetPlayerActions`` on each
simulation cycle, sending a :class:`State` message with the current
world-model and expecting back :class:`PlayerActions`.

This module provides :class:`GameServicer`, a :mod:`grpc` servicer that
bridges between the sidecar and the RL training loop.  It is intended to
be started by :class:`~rcss_rl.env.rcss_env.RCSSEnv` when running in
*remote* mode (i.e. connected to an actual rcss_cluster room).

A single ``GameServicer`` handles **all** training agents in a match.
Each agent's sidecar calls ``GetPlayerActions`` concurrently; the
servicer dispatches by ``world_model.self.id`` (uniform number).

The servicer uses **asyncio queues** for state/action exchange so that
``GetPlayerActions`` is a native ``async def`` coroutine compatible with
``grpc.aio``.  The :func:`serve` helper runs the async gRPC server on a
dedicated background thread with its own event loop.  Synchronous callers
(e.g. the Gymnasium :class:`~rcss_rl.env.rcss_env.RCSSEnv`) interact via
:meth:`~GameServicer.wait_for_state` / :meth:`~GameServicer.set_actions`
which schedule coroutines onto that loop using
:func:`asyncio.run_coroutine_threadsafe`.

Usage sketch (called internally by *RCSSEnv*)::

    from rcss_rl.env.grpc_service import GameServicer, serve

    servicer = GameServicer()
    servicer.register_player(2)
    servicer.register_player(3)
    server, loop = serve(servicer, port=50051, block=False)

Dependencies
------------
``grpcio`` and ``grpcio-tools`` must be installed.  The Python stubs
have been pre-generated from ``rcss_rl/proto/service.proto``.
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

STATE_GET_TIMEOUT_S = 2
STATE_SEND_TIMEOUT_S = 2
ACTION_GET_TIMEOUT_S = 30
ACTION_SEND_TIMEOUT_S = 2

class GameServicer(pb2_grpc.GameServicer):
    """Async gRPC servicer implementing the ``Game`` service.

    The servicer maintains **per-unum** async queues for state/action
    exchange.  When a sidecar calls ``GetPlayerActions``, the coroutine:

    1. Puts the incoming :class:`pb2.State` onto ``_state_queues[unum]``.
    2. Awaits a :class:`pb2.PlayerActions` from ``_action_queues[unum]``.

    The training loop consumes states via :meth:`wait_for_state` and
    provides actions via :meth:`set_actions`.  Both methods are
    **synchronous wrappers** that schedule work onto the servicer's
    event loop (set via :meth:`bind_loop`) using
    :func:`asyncio.run_coroutine_threadsafe`, making them safe to call
    from the Gymnasium thread.

    Call :meth:`register_player` for each training agent **before**
    starting the gRPC server.  Only registered unums participate in the
    synchronized state/action exchange.
    """

    def __init__(self) -> None:
        # Per-unum asyncio queues (maxsize=1 for lock-step exchange).
        self._batcher: BatchQueue[pb2.State] = BatchQueue()
        self._states_queues: dict[int, asyncio.Queue[tuple[int, pb2.State]]] = {}
        self._action_queues: dict[int, asyncio.Queue[pb2.PlayerActions]] = {}

        # Latest state cache — populated by wait_for_state so that
        # get_state / get_world_model remain cheap synchronous reads.
        self._states: dict[int, pb2.State] = {}

        self._server_params: pb2.ServerParam | None = None
        self._player_params: pb2.PlayerParam | None = None
        self._player_types: dict[int, pb2.PlayerType] = {}
        self._debug_mode: bool = False

        # The asyncio event loop that owns the queues.  Set by
        # :meth:`bind_loop` (called automatically by :func:`serve`).
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---- Event-loop binding -----------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the servicer to an asyncio event loop.

        Must be called from the thread that owns *loop* **before** any
        RPC arrives, so that the queues are created on the correct loop.
        Existing queues are recreated.
        """
        self._loop = loop
        # (Re)create queues on this loop.
        self.reset()

    # ---- Player registration ---------------------------------------------

    def register(self, unum: int) -> None:
        """Register a training agent by uniform number.

        Must be called before the gRPC server begins serving so that the
        per-unum queues are ready when the first ``GetPlayerActions``
        arrives.  If the event loop is already bound, the queues are
        created immediately; otherwise they are created when
        :meth:`bind_loop` is called.
        """
        self._states_queues[unum] = asyncio.Queue(maxsize=1)
        self._action_queues[unum] = asyncio.Queue(maxsize=1)
        self._batcher.register(unum, self._states_queues[unum])

    def unregister(self, unum: int) -> None:
        """Unregister a training agent by uniform number."""
        self._batcher.unregister(unum)
        self._states_queues.pop(unum, None)
        self._action_queues.pop(unum, None)
        self._states.pop(unum, None)

    @property
    def unums(self) -> frozenset[int]:
        """Return the set of registered uniform numbers."""
        return self._batcher.unums()

    # ---- RPC handlers (called by the SoccerSimulationProxy sidecar) ------

    async def __get_action(self, unum: int, state: pb2.State) -> pb2.PlayerActions | None:
        """Internal helper to handle GetPlayerActions logic for a single unum."""
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
        """Receive world state from sidecar and return player actions.

        This is an ``async def`` handler compatible with ``grpc.aio``.
        The sidecar is identified by ``request.world_model.self.id``
        (unum).  If the unum is not registered, an empty
        :class:`pb2.PlayerActions` is returned immediately.
        """
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
        """Handle coach state — returns empty actions by default."""
        logger.debug("GetCoachActions called (cycle=%d)", request.world_model.cycle)
        return pb2.CoachActions()

    async def GetTrainerActions(
        self, request: pb2.State, context: grpc.aio.ServicerContext
    ) -> pb2.TrainerActions:
        """Handle trainer state — returns empty actions by default."""
        logger.debug("GetTrainerActions called (cycle=%d)", request.world_model.cycle)
        return pb2.TrainerActions()

    async def SendInitMessage(
        self, request: pb2.InitMessage, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Receive initialisation message from sidecar."""
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
        """Store server parameters sent by the sidecar."""
        self._server_params = request
        logger.debug("Received ServerParam")
        return pb2.Empty()

    async def SendPlayerParams(
        self, request: pb2.PlayerParam, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Store player parameters sent by the sidecar."""
        self._player_params = request
        logger.debug("Received PlayerParam")
        return pb2.Empty()

    async def SendPlayerType(
        self, request: pb2.PlayerType, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Store player type information sent by the sidecar.

        Player types are keyed by their ``id`` field, following the
        reference implementation pattern.
        """
        self._player_types[request.id] = request
        logger.debug("Received PlayerType id=%d", request.id)
        return pb2.Empty()

    async def GetInitMessage(
        self, request: pb2.Empty, context: grpc.aio.ServicerContext
    ) -> pb2.InitMessageFromServer:
        """Return init message to sidecar."""
        return pb2.InitMessageFromServer()

    # ---- Async interface for the training loop ----------------------------

    async def __fetch_states(self, timeout: float) -> dict[int, pb2.State]:
        """Await until *unum*'s state arrives.  Returns *True* on success.

        The received state is cached internally so that subsequent calls
        to :meth:`get_state` / :meth:`get_world_model` return it.
        """
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
        """Provide actions for *unum* and wake its ``GetPlayerActions``."""
        action_q = self._action_queues.get(unum)
        if action_q is None: raise RuntimeError(f"Player unum={unum} not registered")
        await action_q.put(action)

    async def __send_actions(
        self, actions: dict[int, pb2.PlayerActions]
    ) -> set[int]:
        """Provide actions for multiple unums at once."""
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


    # ---- Sync wrappers (for Gymnasium / threading callers) ----------------

    def __run_coro(self, coro: Any, timeout: float | None = None) -> Any:
        """Schedule *coro* on the bound event loop and block for the result.

        Raises :class:`RuntimeError` if no loop has been bound yet.
        """
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
        """Block until all registered agents' states arrive.

        Returns a dict mapping unum → State.
        Thread-safe synchronous wrapper around :meth:`__get_states`.
        """
        return self.__run_coro(
            self.__fetch_states(timeout=timeout),
            timeout=timeout+0.5,
        )

    def send_action(self, unum: int, action: pb2.PlayerActions) -> None:
        """Provide actions for *unum*.

        Thread-safe synchronous wrapper around :meth:`async_set_actions`.
        """
        self.__run_coro(self.__send_action(unum, action), timeout=ACTION_SEND_TIMEOUT_S)

    def send_actions(self, actions: dict[int, pb2.PlayerActions]) -> None:
        """Provide actions for multiple unums at once."""
        self.__run_coro(self.__send_actions(actions), timeout=ACTION_SEND_TIMEOUT_S)

    # ---- Synchronous read helpers (no loop needed) -----------------------

    def last_state(self, unum: int) -> pb2.State | None:
        """Return the latest :class:`pb2.State` for *unum*."""
        return self._states.get(unum)

    def server_params(self) -> pb2.ServerParam | None:
        """Return stored :class:`pb2.ServerParam` message."""
        return self._server_params

    def player_params(self) -> pb2.PlayerParam | None:
        """Return stored :class:`pb2.PlayerParam` message."""
        return self._player_params

    def player_type(self, type_id: int) -> pb2.PlayerType | None:
        """Return the :class:`pb2.PlayerType` with the given *type_id*."""
        return self._player_types.get(type_id)

    def player_types(self) -> dict[int, pb2.PlayerType]:
        """Return a copy of all stored player types."""
        return dict(self._player_types)

    def reset(self) -> None:
        """Clear all per-unum state/action buffers and queues.

        Call between episodes to prepare for fresh state arrivals.
        Registration is preserved.  Queues are drained and recreated.
        """
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

def _run_aio_server(
    servicer: GameServicer,
    port: int,
    started: threading.Event,
    server_holder: list,
    block: bool,
) -> None:
    """Entry-point for the background thread that runs the async gRPC server."""
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

    # Bind the servicer to this loop so sync wrappers work.
    servicer.bind_loop(loop)

    server_holder.append(server)
    started.set()

    if block:
        loop.run_until_complete(server.wait_for_termination())
    else:
        # Keep the loop running so coroutines can be scheduled on it.
        loop.run_forever()


def serve(
    servicer: GameServicer | None = None,
    port: int = 50051,
    block: bool = True,
) -> tuple[grpc_aio.Server, asyncio.AbstractEventLoop]:
    """Start the async gRPC server on a dedicated background thread.

    Parameters
    ----------
    servicer:
        The :class:`GameServicer` instance.  A new one is created if *None*.
    port:
        TCP port to listen on.
    block:
        If *True*, blocks the calling thread until the server is terminated.
        If *False*, returns immediately — the server (and its event loop)
        keeps running on a daemon thread.

    Returns
    -------
    tuple[grpc_aio.Server, asyncio.AbstractEventLoop]
        The running server and the event loop it runs on.
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

    # Wait until the server is actually listening.
    started.wait(timeout=30.0)
    if not server_holder:
        raise RuntimeError("gRPC aio server failed to start")

    server = server_holder[0]
    loop = servicer._loop
    assert loop is not None, "Loop should have been bound by _run_aio_server"

    if block:
        thread.join()

    return server, loop
