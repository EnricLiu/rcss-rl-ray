from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .info import FleetInfo

if TYPE_CHECKING:
    from ..base import AllocatorClient


class FleetClient:
    def __init__(self, info: FleetInfo | dict, client: AllocatorClient) -> None:
        if isinstance(info, dict):
            info = FleetInfo.model_validate(info)

        self.__info = cast(FleetInfo, info)
        self.__client = client

    @property
    def info(self) -> FleetInfo:
        return self.__info

    @property
    def client(self) -> AllocatorClient:
        return self.__client

    @property
    def name(self) -> str:
        return self.info.name

    def drop(self) -> None:
        self.client.drop_fleet(self.name)

    @property
    def template(self) -> dict:
        return self.client.fleet_get_template(fmt="json")

    @property
    def template_version(self) -> str:
        return self.client.fleet_get_template_version()
