"""REST client for the rcss_cluster match allocator.

The allocator is a Kubernetes-hosted service that manages simulation
rooms.  Clients request a room via ``POST /rooms`` with a JSON body
that describes team composition, stopping conditions, etc.  The
allocator provisions the room and returns a connection address.

This module provides :class:`AllocatorClient` as a thin wrapper around
the allocator's HTTP API.  It is used by
:class:`~rcss_rl.env.rcss_env.RCSSEnv` when running in *remote* mode.

Dependencies
------------
Requires the ``requests`` library (listed as an optional dependency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlayerConfig:
    """Configuration for a single player in a match.

    Attributes
    ----------
    unum : int
        Uniform number (1-11).
    goalie : bool
        Whether this player is a goalie.
    policy_kind : str
        ``"bot"`` for a scripted agent or ``"agent"`` for an RL training
        agent.
    policy_image : str | None
        Docker image for bot players (e.g. ``"HELIOS/helios-base"``).
        Ignored when *policy_kind* is ``"agent"``.
    """

    unum: int = 1
    goalie: bool = False
    policy_kind: str = "agent"
    policy_image: str | None = None


@dataclass
class RoomRequest:
    """Payload for requesting a simulation room from the allocator.

    Mirrors the structure of the ``template.json`` accepted by the
    rcss_cluster match-composer sidecar.

    Attributes
    ----------
    ally_name : str
        Name for the allied (training) team.
    opponent_name : str
        Name for the opponent team.
    ally_players : list[PlayerConfig]
        Player specifications for the allied team.
    opponent_players : list[PlayerConfig]
        Player specifications for the opponent team.
    time_up : int
        Simulation stop time (in server cycles).
    grpc_host : str
        Host address of the training-side gRPC server (reachable by the
        sidecar).
    grpc_port : int
        Port of the training-side gRPC server.
    """

    ally_name: str = "RLAgent"
    opponent_name: str = "Bot"
    ally_players: list[PlayerConfig] = field(default_factory=list)
    opponent_players: list[PlayerConfig] = field(default_factory=list)
    time_up: int = 6000
    grpc_host: str = "localhost"
    grpc_port: int = 50051

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON structure expected by the allocator."""
        def _player(p: PlayerConfig) -> dict[str, Any]:
            policy: dict[str, Any] = {"kind": p.policy_kind}
            if p.policy_image:
                policy["image"] = p.policy_image
            return {
                "unum": p.unum,
                "goalie": p.goalie,
                "policy": policy,
            }

        return {
            "api_version": 1,
            "stopping": {"time_up": self.time_up},
            "teams": {
                "allies": {
                    "name": self.ally_name,
                    "players": [_player(p) for p in self.ally_players],
                },
                "opponents": {
                    "name": self.opponent_name,
                    "players": [_player(p) for p in self.opponent_players],
                },
            },
        }


class AllocatorClient:
    """HTTP client for the rcss_cluster room allocator.

    Parameters
    ----------
    base_url:
        Allocator endpoint, e.g. ``"http://allocator.rcss.svc:6000"``.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def request_room(self, room_request: RoomRequest) -> dict[str, Any]:
        """Ask the allocator for a simulation room.

        Parameters
        ----------
        room_request:
            The room configuration payload.

        Returns
        -------
        dict
            JSON response from the allocator containing the room
            connection address and metadata.

        Raises
        ------
        ImportError
            If the ``requests`` library is not installed.
        RuntimeError
            If the allocator returns a non-2xx status code.
        """
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required for remote mode. "
                "Install it with: pip install requests"
            ) from exc

        url = f"{self._base_url}/rooms"
        payload = room_request.to_dict()

        logger.info("Requesting room from %s", url)
        logger.debug("Payload: %s", payload)

        resp = requests.post(url, json=payload, timeout=self._timeout)
        if not resp.ok:
            raise RuntimeError(
                f"Allocator returned {resp.status_code}: {resp.text}"
            )

        data: dict[str, Any] = resp.json()
        logger.info("Room allocated: %s", data)
        return data

    def release_room(self, room_id: str) -> None:
        """Release (delete) a previously allocated room.

        Parameters
        ----------
        room_id:
            Identifier of the room to release.
        """
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required for remote mode. "
                "Install it with: pip install requests"
            ) from exc

        url = f"{self._base_url}/rooms/{room_id}"
        logger.info("Releasing room %s via %s", room_id, url)

        resp = requests.delete(url, timeout=self._timeout)
        if not resp.ok:
            logger.warning("Failed to release room %s: %s", room_id, resp.text)

    def health_check(self) -> bool:
        """Return *True* if the allocator is reachable."""
        try:
            import requests
        except ImportError:
            return False

        try:
            resp = requests.get(
                f"{self._base_url}/health", timeout=self._timeout
            )
            return resp.ok
        except Exception:
            return False
