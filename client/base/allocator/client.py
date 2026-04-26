"""Room allocator REST client.

Communicates with the rcss_cluster allocator over HTTP.  The route layout
mirrors the Rust backend (``ref/controller``):

    POST   /gs/allocate           — allocate a simulation room (GameServer)
    POST   /fleet/create          — create a fleet
    DELETE /fleet/                 — drop a fleet (JSON body)
    GET    /fleet/template         — fleet CRD template
    GET    /fleet/template/version — fleet template version
    GET    /health                 — liveness check
    GET    /ready                  — readiness check

All responses are wrapped in a standardized envelope::

    Success: {"data": <payload>}
    Error:   {"error": "<message>"}
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from httpx import Client, Response

from ..http import BaseApiClient, unwrap_response
from ...fleet import FleetClient, FleetInfo
from ...room import RoomClient
from utils import retry
from schema import GameServerSchema, SCHEMA_VERSION
from .config import AllocatorConfig

from .model import (
    DeleteDropFleetRequest,
    GetFleetTemplateRequest,
    PostAllocateRoomRequest,
    PostCreateFleetRequest,
)

logger = logging.getLogger(__name__)

class AllocatorClient(BaseApiClient):
    """HTTP client for the rcss_cluster allocator."""

    def __init__(
        self,
        config: AllocatorConfig,
        client: Client | None = None,
    ) -> None:
        self.__cfg = config
        super().__init__(config, client=client)

    @property
    def config(self) -> AllocatorConfig:
        return self.__cfg

    @property
    def base_url(self) -> str:
        return self.__cfg.base_url

    @property
    def timeout(self) -> float:
        return self.__cfg.timeout_s

    # ---- Response helpers ------------------------------------------------

    @staticmethod
    def unwrap_response(resp: Response) -> dict[str, Any]:
        return unwrap_response(resp, expect_envelope=True)

    @retry(max_retries=3, delay=0.5, backoff=1.0, logger=logger)
    def request_room(
        self,
        schema: GameServerSchema,
        fleet: str | None = None,
        version: int = SCHEMA_VERSION,
    ) -> RoomClient:
        if fleet is not None:
            raise ValueError(
                "Current allocator /gs/allocate API does not support selecting a fleet explicitly"
            )

        payload = PostAllocateRoomRequest(
            conf=schema,
            version=version,
        )

        data = self._request_payload(
            "POST",
            self.config.path_room_alloc,
            json=payload,
        )

        room = RoomClient(data, self)
        return room

    @retry(max_retries=3, delay=0.5, backoff=1.0, logger=logger)
    def create_fleet(
        self,
        name: str,
        schema: GameServerSchema,
        version: int = SCHEMA_VERSION,
    ) -> FleetClient:
        payload = PostCreateFleetRequest(
            name=name,
            conf=schema,
            version=version,
        )

        self._request_payload(
            "POST",
            self.config.path_fleet_create,
            json=payload,
        )

        return FleetClient(FleetInfo(name=name), self)

    @retry(max_retries=3, delay=0.5, backoff=1.0, logger=logger)
    def drop_fleet(self, fleet_name: str) -> None:
        payload = DeleteDropFleetRequest(
            name=fleet_name,
        )

        self._request_payload(
            "DELETE",
            self.config.path_fleet_drop,
            json=payload,
        )

    def fleet_get_template(self, fmt: Literal["json", "yaml"] = "json") -> Any:
        params = GetFleetTemplateRequest(format=fmt)

        payload = self._request_payload(
            "GET",
            self.config.path_fleet_template,
            params=params.model_dump(mode="json", by_alias=True),
        )

        if isinstance(payload, dict) and "template" in payload:
            return payload["template"]
        return payload

    def fleet_get_template_version(self) -> str:
        payload = self._request_payload(
            "GET",
            self.config.path_fleet_template_version,
        )
        if isinstance(payload, dict):
            return str(payload.get("version", ""))
        return str(payload)

    def health_check(self) -> bool:
        """Liveness check — ``GET /health``. Returns ``True`` if healthy."""
        resp = self.client.get("/health")
        return resp.is_success

    def readiness_check(self) -> bool:
        """Readiness check — ``GET /ready``. Returns ``True`` if ready."""
        resp = self.client.get("/ready")
        return resp.is_success