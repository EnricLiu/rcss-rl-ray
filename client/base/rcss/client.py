from __future__ import annotations

from typing import Any, Literal

from httpx import Client

from ..http import BaseApiClient

from .config import RcssConfig, TrainerConfig
from .model import (
    ControlShutdownRequest,
    MetricsConnectionInfo,
    MetricsStatusResponse,
    TrainerChangeModeRequest,
    TrainerCheckBallResponse,
    TrainerCommandResult,
    TrainerEarRequest,
    TrainerInitRequest,
    TrainerTeamNamesResponse,
)


class RcssTrainerClient:
    def __init__(self, config: TrainerConfig, api: BaseApiClient) -> None:
        self._api = api
        self.__config = config

    @property
    def config(self) -> TrainerConfig:
        return self.__config

    def _command(self, path: str, payload: Any | None = None) -> Any:
        body = payload if payload is not None else dict()
        return self._api._request_payload("POST", path, json=body, )

    def change_mode(self, play_mode: str) -> TrainerCommandResult:
        payload = self._command(
            self.config.path_change_mode,
            TrainerChangeModeRequest(play_mode=play_mode).model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            ),
        )
        return TrainerCommandResult.model_validate(payload)

    def check_ball(self) -> TrainerCheckBallResponse:
        payload = self._command(self.config.path_check_ball)
        return TrainerCheckBallResponse.model_validate(payload)

    def ear(self, mode: Literal["on", "off"]) -> TrainerCommandResult:
        payload = self._command(
            self.config.path_ear,
            TrainerEarRequest(mode=mode).model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            ),
        )
        return TrainerCommandResult.model_validate(payload)

    def eye(self, mode: Literal["on", "off"]) -> TrainerCommandResult:
        payload = self._command(
            self.config.path_eye,
            TrainerEarRequest(mode=mode).model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            ),
        )
        return TrainerCommandResult.model_validate(payload)

    def init(self, version: int | None = None) -> TrainerCommandResult:
        payload = self._command(
            self.config.path_init,
            TrainerInitRequest(version=version).model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            ),
        )
        return TrainerCommandResult.model_validate(payload)

    def look(self) -> TrainerCommandResult:
        payload = self._command(self.config.path_look)
        return TrainerCommandResult.model_validate(payload)

    def move(self, payload: dict[str, Any] | None = None) -> TrainerCommandResult:
        response = self._command(self.config.path_move, payload or {"todo": None})
        return TrainerCommandResult.model_validate(response)

    def recover(self) -> TrainerCommandResult:
        payload = self._command(self.config.path_recover)
        return TrainerCommandResult.model_validate(payload)

    def start(self) -> TrainerCommandResult:
        payload = self._command(self.config.path_start)
        return TrainerCommandResult.model_validate(payload)

    def team_names(self) -> TrainerTeamNamesResponse:
        payload = self._command(self.config.path_team_names)
        return TrainerTeamNamesResponse.model_validate(payload)


class RcssClient(BaseApiClient):
    def __init__(
        self,
        config: RcssConfig | str,
        *,
        timeout: float = 10,
        client: Client | None = None,
    ) -> None:
        if isinstance(config, str):
            cfg: RcssConfig = RcssConfig(base_url=config, timeout_s=timeout)
        else:
            cfg = config

        self.__config: RcssConfig = cfg
        super().__init__(cfg, client=client)
        self._trainer = RcssTrainerClient(cfg.trainer, self)

    @property
    def config(self) -> RcssConfig:
        return self.__config

    @property
    def trainer(self) -> RcssTrainerClient:
        return self._trainer

    def shutdown(self, *, force: bool = True) -> None:
        self._request_payload(
            "POST",
            self.config.control.path_shutdown,
            json=ControlShutdownRequest(force=force),
        )

    def metrics_status(self) -> MetricsStatusResponse:
        payload = self._request_payload("GET", self.config.metrics.path_status)
        return MetricsStatusResponse.model_validate(payload)

    def metrics_health(self) -> MetricsStatusResponse:
        payload = self._request_payload("GET", self.config.metrics.path_health)
        return MetricsStatusResponse.model_validate(payload)

    def metrics_conn(self) -> dict[str, MetricsConnectionInfo]:
        payload = self._request_payload("GET", self.config.metrics.path_conn)
        status = payload.get("status", {}) if isinstance(payload, dict) else {}
        return {
            key: MetricsConnectionInfo.model_validate(value)
            for key, value in status.items()
        }
