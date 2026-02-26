"""gRPC ``Game`` service implementation for SoccerSimulationProxy.

The SoccerSimulationProxy sidecar calls ``GetPlayerActions`` on each
simulation cycle, sending a :class:`State` message with the current
world-model and expecting back :class:`PlayerActions`.

This module provides :class:`GameServicer`, a :mod:`grpc` servicer that
bridges between the sidecar and the RL training loop.  It is intended to
be started by :class:`~rcss_rl.env.rcss_env.RCSSEnv` when running in
*remote* mode (i.e. connected to an actual rcss_cluster room).

Usage sketch (called internally by *RCSSEnv*)::

    server = grpc.server(futures.ThreadPoolExecutor())
    servicer = GameServicer()
    game_pb2_grpc.add_GameServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()

Dependencies
------------
``grpcio`` and ``grpcio-tools`` must be installed, and the Python stubs
must be generated from ``rcss_rl/proto/service.proto`` before importing
this module::

    python -m grpc_tools.protoc \\
        -I rcss_rl/proto \\
        --python_out=rcss_rl/proto \\
        --grpc_python_out=rcss_rl/proto \\
        rcss_rl/proto/service.proto
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class GameServicer:
    """gRPC servicer implementing the ``Game`` service from *service.proto*.

    The servicer maintains a shared state buffer that is written by the
    sidecar (via ``GetPlayerActions``) and consumed by the environment's
    ``step()`` method.  A pair of :class:`threading.Event` objects is used
    to synchronise the two sides.

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

        self._current_state: Any | None = None
        self._current_actions: Any | None = None
        self._server_params: Any | None = None
        self._player_params: Any | None = None
        self._player_types: list[Any] = []
        self._lock = threading.Lock()

    # ---- RPC handlers ----------------------------------------------------

    def GetPlayerActions(self, request: Any, context: Any) -> Any:
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
            return self._current_actions

    def GetCoachActions(self, request: Any, context: Any) -> Any:
        """Handle coach state (currently returns empty actions)."""
        logger.debug("GetCoachActions called (no-op)")
        return None  # Placeholder

    def GetTrainerActions(self, request: Any, context: Any) -> Any:
        """Handle trainer state (currently returns empty actions)."""
        logger.debug("GetTrainerActions called (no-op)")
        return None  # Placeholder

    def SendInitMessage(self, request: Any, context: Any) -> Any:
        """Receive initialisation message from sidecar."""
        logger.info("Received InitMessage: agent_type=%s", getattr(request, "agent_type", "?"))
        return None  # Placeholder

    def SendServerParams(self, request: Any, context: Any) -> Any:
        """Store server parameters sent by the sidecar."""
        with self._lock:
            self._server_params = request
        logger.debug("Received ServerParam")
        return None  # Placeholder

    def SendPlayerParams(self, request: Any, context: Any) -> Any:
        """Store player parameters sent by the sidecar."""
        with self._lock:
            self._player_params = request
        logger.debug("Received PlayerParam")
        return None  # Placeholder

    def SendPlayerType(self, request: Any, context: Any) -> Any:
        """Store player type information sent by the sidecar."""
        with self._lock:
            self._player_types.append(request)
        logger.debug("Received PlayerType id=%s", getattr(request, "id", "?"))
        return None  # Placeholder

    def GetInitMessage(self, request: Any, context: Any) -> Any:
        """Return init message to sidecar (placeholder)."""
        return None  # Placeholder

    # ---- Interface for the training loop ---------------------------------

    def get_state(self) -> Any | None:
        """Return the latest ``State`` message (or *None*)."""
        with self._lock:
            return self._current_state

    def set_actions(self, actions: Any) -> None:
        """Provide the ``PlayerActions`` reply and wake the sidecar."""
        with self._lock:
            self._current_actions = actions
        self.action_ready.set()

    def get_server_params(self) -> Any | None:
        """Return stored ``ServerParam`` message."""
        with self._lock:
            return self._server_params
