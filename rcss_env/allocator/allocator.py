"""Room allocator REST client.

Communicates with the rcss_cluster allocator over HTTP:
  POST   /rooms        — request a simulation room
  DELETE  /rooms/{id}   — release a simulation room
  GET    /health       — health check
"""

from __future__ import annotations

from typing import Any
from schema import GameServerSchema
from config import AllocatorConfig
import logging

import httpx

logger = logging.getLogger(__name__)


class AllocatorClient:
    """HTTP client for the rcss_cluster room allocator."""

    def __init__(self, config: AllocatorConfig) -> None:
        self.__cfg = config

    @property
    def base_url(self) -> str:
        return self.__cfg.base_url

    @property
    def timeout(self) -> float:
        return self.__cfg.timeout_s

    def request_room(self, schema: GameServerSchema) -> dict[str, Any]:
        """Request a simulation room from the allocator.

        Serializes the GameServerSchema to JSON and POSTs it to the allocator.
        Returns the allocator's JSON response dict (expected to contain ``room_id``, etc.).

        Raises:
            RuntimeError: The allocator returned a non-2xx status code.
        """
        url = f"{self.base_url}/rooms"
        payload = schema.model_dump(mode="json", by_alias=True)

        logger.info("Requesting room from %s", url)
        logger.debug("Payload: %s", payload)

        resp = httpx.post(url, json=payload, timeout=self.timeout)
        if resp.is_error:
            raise RuntimeError(
                f"Allocator returned {resp.status_code}: {resp.text}"
            )

        data: dict[str, Any] = resp.json()
        logger.info("Room allocated: %s", data)
        return data

    def release_room(self, room_id: str) -> None:
        """Release a previously allocated simulation room.

        Args:
            room_id: ID of the room to release.
        """
        url = f"{self.base_url}/rooms/{room_id}"
        logger.info("Releasing room %s via %s", room_id, url)

        resp = httpx.delete(url, timeout=self.timeout)
        if resp.is_error:
            logger.warning("Failed to release room %s: %s", room_id, resp.text)

    def health_check(self) -> bool:
        """Check whether the allocator is reachable. Returns True if healthy."""
        try:
            resp = httpx.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
            return resp.is_success

        except Exception:
            return False
