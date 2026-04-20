from __future__ import annotations

from typing import Any, Literal, override

from httpx import Client

from client.base.http import BaseApiClient, dump_json_payload
from .model import MatchStatusResponse, MatchTeamInfo
from .config import MatchComposerConfig


class MatchComposerClient(BaseApiClient):
    def __init__(
        self,
        config: MatchComposerConfig,
        client: Client | None = None,
    ) -> None:
        super().__init__(config, client=client)

    @override
    @property
    def config(self) -> MatchComposerConfig:
        return self.__config

    # def start(self, config: Any | None = None) -> None:
    #     self._request_payload("POST", self.config.path_match_start, json=dump_json_payload(config))
    #
    # def stop(self) -> None:
    #     self._request_payload("POST", self.config.path_match_stop)
    #
    # def restart(self, config: Any | None = None) -> None:
    #     self._request_payload("POST", self.config.path_match_restart, json=dump_json_payload(config))

    def status(self) -> MatchStatusResponse:
        payload = self._request_payload("GET", self.config.path_match_status)
        return MatchStatusResponse.model_validate(payload)

    def team_status(self, side: Literal["left", "right"]) -> MatchTeamInfo:
        payload = self._request_payload(
            "GET",
            self.config.path_team_status,
            params={"side": side},
        )
        return MatchTeamInfo.model_validate(payload)