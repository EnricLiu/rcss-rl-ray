"""gRPC ``Game`` service implementation for SoccerSimulationProxy.

The SoccerSimulationProxy sidecar calls ``GetPlayerActions`` on each
simulation cycle, sending a :class:`State` message with the current
world-model and expecting back :class:`PlayerActions`.

This module provides :class:`GameServicer`, a :mod:`grpc` servicer that
bridges between the sidecar and the RL training loop.  It is intended to
be started by :class:`~rcss_rl.env.rcss_env.RCSSEnv` when running in
*remote* mode (i.e. connected to an actual rcss_cluster room).

The design follows the reference implementation from
`PlaymakerServer-Python <https://github.com/Cyrus2D/PlaymakerServer-Python>`_.

Usage sketch (called internally by *RCSSEnv*)::

    from rcss_rl.env.grpc_service import GameServicer, serve

    servicer = GameServicer()
    server = serve(servicer, port=50051, block=False)

Dependencies
------------
``grpcio`` and ``grpcio-tools`` must be installed.  The Python stubs
have been pre-generated from ``rcss_rl/proto/service.proto``.
"""

from __future__ import annotations

import logging
import threading
from concurrent import futures
from typing import Any

import grpc

from rcss_rl.proto import service_pb2 as pb2
from rcss_rl.proto import service_pb2_grpc

logger = logging.getLogger(__name__)


class GameServicer(service_pb2_grpc.GameServicer):
    """gRPC servicer implementing the ``Game`` service from *service.proto*.

    The servicer maintains a shared state buffer that is written by the
    sidecar (via ``GetPlayerActions``) and consumed by the environment's
    ``step()`` method.  A pair of :class:`threading.Event` objects is used
    to synchronise the two sides.

    Following the SoccerSimulationProxy protocol:

    * ``SendServerParams``, ``SendPlayerParams``, ``SendPlayerType``, and
      ``SendInitMessage`` are called during initialisation to deliver
      simulation parameters.  All return :class:`pb2.Empty`.
    * ``GetInitMessage`` is called by the sidecar to retrieve any init
      configuration.  Returns :class:`pb2.InitMessageFromServer`.
    * ``GetPlayerActions`` is called each cycle with the current
      :class:`pb2.State`.  It blocks until the training loop provides a
      :class:`pb2.PlayerActions` via :meth:`set_actions`.
    * ``GetCoachActions`` and ``GetTrainerActions`` return empty
      action lists by default.

    Attributes
    ----------
    state_ready : threading.Event
        Set when a new ``State`` message has been received.
    action_ready : threading.Event
        Set when the training loop has produced a ``PlayerActions`` reply.
    """

    def __init__(self) -> None:
        self.state_ready = threading.Event()
        self.action_ready = threading.Event()

        self._current_state: pb2.State | None = None
        self._current_actions: pb2.PlayerActions | None = None
        self._server_params: pb2.ServerParam | None = None
        self._player_params: pb2.PlayerParam | None = None
        self._player_types: dict[int, pb2.PlayerType] = {}
        self._debug_mode: bool = False
        self._lock = threading.Lock()

    # ---- RPC handlers (called by the SoccerSimulationProxy sidecar) ------

    def GetPlayerActions(
        self, request: pb2.State, context: grpc.ServicerContext
    ) -> pb2.PlayerActions:
        """Receive world state from sidecar and return player actions.

        This method blocks until the training loop provides actions via
        :meth:`set_actions`.
        """
        with self._lock:
            self._current_state = request
        self.state_ready.set()

        # Wait for the training loop to compute actions.
        self.action_ready.wait()
        self.action_ready.clear()

        with self._lock:
            actions = self._current_actions
            self._current_actions = None
        return actions if actions is not None else pb2.PlayerActions()

    def GetCoachActions(
        self, request: pb2.State, context: grpc.ServicerContext
    ) -> pb2.CoachActions:
        """Handle coach state — returns empty actions by default."""
        logger.debug("GetCoachActions called (cycle=%d)", request.world_model.cycle)
        return pb2.CoachActions()

    def GetTrainerActions(
        self, request: pb2.State, context: grpc.ServicerContext
    ) -> pb2.TrainerActions:
        """Handle trainer state — returns empty actions by default."""
        logger.debug("GetTrainerActions called (cycle=%d)", request.world_model.cycle)
        return pb2.TrainerActions()

    def SendInitMessage(
        self, request: pb2.InitMessage, context: grpc.ServicerContext
    ) -> pb2.Empty:
        """Receive initialisation message from sidecar."""
        self._debug_mode = request.debug_mode
        logger.info(
            "Received InitMessage: agent_type=%s, debug_mode=%s",
            request.agent_type,
            request.debug_mode,
        )
        return pb2.Empty()

    def SendServerParams(
        self, request: pb2.ServerParam, context: grpc.ServicerContext
    ) -> pb2.Empty:
        """Store server parameters sent by the sidecar."""
        with self._lock:
            self._server_params = request
        logger.debug("Received ServerParam")
        return pb2.Empty()

    def SendPlayerParams(
        self, request: pb2.PlayerParam, context: grpc.ServicerContext
    ) -> pb2.Empty:
        """Store player parameters sent by the sidecar."""
        with self._lock:
            self._player_params = request
        logger.debug("Received PlayerParam")
        return pb2.Empty()

    def SendPlayerType(
        self, request: pb2.PlayerType, context: grpc.ServicerContext
    ) -> pb2.Empty:
        """Store player type information sent by the sidecar.

        Player types are keyed by their ``id`` field, following the
        reference implementation pattern.
        """
        with self._lock:
            self._player_types[request.id] = request
        logger.debug("Received PlayerType id=%d", request.id)
        return pb2.Empty()

    def GetInitMessage(
        self, request: pb2.Empty, context: grpc.ServicerContext
    ) -> pb2.InitMessageFromServer:
        """Return init message to sidecar."""
        return pb2.InitMessageFromServer()

    # ---- Interface for the training loop ---------------------------------

    def get_state(self) -> pb2.State | None:
        """Return the latest :class:`pb2.State` message (or *None*)."""
        with self._lock:
            return self._current_state

    def get_world_model(self) -> pb2.WorldModel | None:
        """Return the :class:`pb2.WorldModel` from the latest state."""
        with self._lock:
            if self._current_state is None:
                return None
            return self._current_state.world_model

    def set_actions(self, actions: pb2.PlayerActions) -> None:
        """Provide the :class:`pb2.PlayerActions` reply and wake the sidecar."""
        with self._lock:
            self._current_actions = actions
        self.action_ready.set()

    def get_server_params(self) -> pb2.ServerParam | None:
        """Return stored :class:`pb2.ServerParam` message."""
        with self._lock:
            return self._server_params

    def get_player_params(self) -> pb2.PlayerParam | None:
        """Return stored :class:`pb2.PlayerParam` message."""
        with self._lock:
            return self._player_params

    def get_player_type(self, type_id: int) -> pb2.PlayerType | None:
        """Return the :class:`pb2.PlayerType` with the given *type_id*."""
        with self._lock:
            return self._player_types.get(type_id)

    def get_player_types(self) -> dict[int, pb2.PlayerType]:
        """Return a copy of all stored player types."""
        with self._lock:
            return dict(self._player_types)


def serve(
    servicer: GameServicer | None = None,
    port: int = 50051,
    max_workers: int = 22,
    block: bool = True,
) -> grpc.Server:
    """Start the gRPC server for the ``Game`` service.

    Parameters
    ----------
    servicer:
        The :class:`GameServicer` instance.  A new one is created if *None*.
    port:
        TCP port to listen on.
    max_workers:
        Maximum number of threads in the gRPC thread pool.
    block:
        If *True*, blocks until the server is terminated.  If *False*,
        returns immediately (useful for embedding in a larger application).

    Returns
    -------
    grpc.Server
        The running gRPC server.
    """
    if servicer is None:
        servicer = GameServicer()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    service_pb2_grpc.add_GameServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC Game server started on port %d", port)

    if block:
        server.wait_for_termination()

    return server
