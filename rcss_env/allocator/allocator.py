from __future__ import annotations

from typing import Any
from schema import RoomSchema
from dataclasses import asdict
import logging

import httpx

logger = logging.getLogger(__name__)

class AllocatorClient:

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def request_room(self, schema: RoomSchema) -> dict[str, Any]:

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

        url = f"{self._base_url}/rooms/{room_id}"
        logger.info("Releasing room %s via %s", room_id, url)

        resp = httpx.delete(url, timeout=self._timeout)
        if resp.is_error:
            logger.warning("Failed to release room %s: %s", room_id, resp.text)

    def health_check(self) -> bool:

        try:
            resp = httpx.get(
                f"{self._base_url}/health", timeout=self._timeout
            )
            return resp.is_success

        except Exception:
            return False
