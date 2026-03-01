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

from typing import Any
from schema import RoomSchema
from dataclasses import asdict
import logging

import httpx

logger = logging.getLogger(__name__)


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

    def request_room(self, schema: RoomSchema) -> dict[str, Any]:
        """Ask the allocator for a simulation room.

        Parameters
        ----------
        schema:
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

        url = f"{self._base_url}/rooms"
        payload = asdict(schema)

        logger.info("Requesting room from %s", url)
        logger.debug("Payload: %s", payload)

        resp = httpx.post(url, json=payload, timeout=self._timeout)
        if resp.is_error:
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

        url = f"{self._base_url}/rooms/{room_id}"
        logger.info("Releasing room %s via %s", room_id, url)

        resp = httpx.delete(url, timeout=self._timeout)
        if resp.is_error:
            logger.warning("Failed to release room %s: %s", room_id, resp.text)

    def health_check(self) -> bool:
        """Return *True* if the allocator is reachable."""
        try:
            resp = httpx.get(
                f"{self._base_url}/health", timeout=self._timeout
            )
            return resp.is_success

        except Exception:
            return False
