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

from rcss_rl.config import PlayerConfig, PlayerInitState

logger = logging.getLogger(__name__)


# Re-export so existing imports from this module keep working.
__all__ = [
    "PlayerInitState",
    "PlayerConfig",
    "RoomRequest",
    "AllocatorClient",
]


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
    goal_limit_l : int
        Stop after the left team scores this many goals (0 = disabled).
    referee_enable : bool
        Whether to enable the referee module.
    ball_init_x : float | None
        Normalised initial X position of the ball (``None`` = default).
    ball_init_y : float | None
        Normalised initial Y position of the ball (``None`` = default).
    """

    ally_name: str = "RLAgent"
    opponent_name: str = "Bot"
    ally_players: list[PlayerConfig] = field(default_factory=list)
    opponent_players: list[PlayerConfig] = field(default_factory=list)
    time_up: int = 6000
    goal_limit_l: int = 0
    referee_enable: bool = False
    ball_init_x: float | None = None
    ball_init_y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON structure expected by the allocator.

        The output conforms to the ``template.json`` format of the
        rcss_cluster match-composer sidecar, including per-player
        ``init_state``, ``blocklist``, and agent policy gRPC fields.
        """

        def _player(p: PlayerConfig) -> dict[str, Any]:
            policy: dict[str, Any] = {"kind": p.policy_kind}
            if p.policy_image:
                policy["image"] = p.policy_image
            if p.policy_kind == "agent":
                if p.policy_agent:
                    policy["agent"] = p.policy_agent
                if p.grpc_host is not None:
                    policy["grpc_host"] = p.grpc_host
                if p.grpc_port is not None:
                    policy["grpc_port"] = p.grpc_port
            result: dict[str, Any] = {
                "unum": p.unum,
                "goalie": p.goalie,
                "policy": policy,
            }
            if p.init_state is not None:
                init_dict = p.init_state.to_dict()
                if init_dict:
                    result["init_state"] = init_dict
            if p.blocklist:
                result["blocklist"] = p.blocklist
            return result

        result: dict[str, Any] = {
            "api_version": 1,
            "referee": {"enable": self.referee_enable},
            "stopping": {"time_up": self.time_up, "goal_l": self.goal_limit_l},
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
        if self.ball_init_x is not None and self.ball_init_y is not None:
            result["init_state"] = {
                "ball": {"x": self.ball_init_x, "y": self.ball_init_y},
            }
        return result


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
