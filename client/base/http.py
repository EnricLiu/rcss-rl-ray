from __future__ import annotations

import logging

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

from httpx import Client, Response
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

class SupportsClientConfig(Protocol):
    base_url: str
    timeout_s: float


class ClusterApiError(RuntimeError):
    pass


class ClusterResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    success: bool
    payload: Any
    created_at: datetime


def dump_json_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json", by_alias=True, exclude_none=True)
    return payload


def _is_envelope(payload: Any) -> bool:
    return isinstance(payload, Mapping) and {
        "id",
        "success",
        "payload",
        "created_at",
    }.issubset(payload)


def _format_error_message(payload: Any) -> str:
    if isinstance(payload, Mapping):
        error = payload.get("error")
        desc = payload.get("desc")
        if error is not None and desc is not None:
            return f"{error}: {desc}"
        if error is not None:
            return str(error)
        if desc is not None:
            return str(desc)
    return str(payload)


def unwrap_response(response: Response, *, expect_envelope: bool = True) -> Any:
    if response.is_error:
        try:
            payload = response.json()
        except ValueError:
            raise ClusterApiError(
                f"HTTP {response.status_code}: {response.text}"
            ) from None

        raise ClusterApiError(
            f"HTTP {response.status_code}: {_format_error_message(payload)}"
        )

    if not response.content:
        return None

    try:
        payload = response.json()
    except ValueError:
        if expect_envelope:
            raise ClusterApiError("Expected JSON response from server") from None
        return response.text

    if not expect_envelope:
        return payload

    if not _is_envelope(payload):
        raise ClusterApiError(
            f"Unexpected response envelope: {payload!r}"
        )

    envelope = ClusterResponse.model_validate(payload)
    if envelope.success:
        return envelope.payload

    raise ClusterApiError(_format_error_message(envelope.payload))


class BaseApiClient:
    def __init__(
        self,
        config: SupportsClientConfig,
        client: Client | None = None,
    ) -> None:
        self.__config = config
        self._client = client or Client(base_url=config.base_url, timeout=config.timeout_s)

    @property
    def config(self) -> SupportsClientConfig:
        return self.__config

    @property
    def client(self) -> Client:
        return self._client

    def close(self) -> None:
        self.client.close()

    def _request_payload(
        self,
        method: str,
        path: str,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        expect_envelope: bool = True,
    ) -> Any:

        logger.info(f"[{method}] -> {self.client.base_url}{path}, json: {json}, params: {params}")
        response = self.client.request(
            method,
            path,
            json=dump_json_payload(json),
            params=params,
        )
        logger.debug(f"[{method}] <- {self.client.base_url}{path}, response: {response.text}")

        res = unwrap_response(response, expect_envelope=expect_envelope)
        logger.info(f"[{method}] <- {self.client.base_url}{path}, resp payload: {res}")

        return res