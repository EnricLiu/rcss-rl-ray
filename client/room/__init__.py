from __future__ import annotations
from typing import TYPE_CHECKING, cast

from ..base import MatchComposerClient, RcssClient
from ..config import ClientConfig
from .info import RoomInfo

if TYPE_CHECKING:
    from ..base import AllocatorClient


class RoomClient:
    __info: RoomInfo
    __client: AllocatorClient
    __mc: MatchComposerClient | None
    __rcss: RcssClient | None

    def __init__(self, info: RoomInfo | dict, client: AllocatorClient) -> None:
        if isinstance(info, dict):
            info = RoomInfo.model_validate(info)

        self.__info = cast(RoomInfo, info)
        self.__client = client
        self.__mc = None
        self.__rcss = None

    @property
    def info(self) -> RoomInfo:
        return self.__info

    @property
    def client(self) -> AllocatorClient:
        return self.__client

    @property
    def rcss(self) -> RcssClient:
        if self.__rcss is None:
            self.__rcss = RcssClient(
                self.info.base_url_rcss,
                timeout=self.client.timeout,
            )
        return self.__rcss

    @property
    def mc(self) -> MatchComposerClient:
        if self.__mc is None:
            self.__mc = MatchComposerClient(
                self.info.base_url_mc,
                timeout=self.client.timeout,
            )
        return self.__mc

    def release(self, *, force: bool = True) -> None:
        self.rcss.shutdown(force=force)