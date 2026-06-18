from __future__ import annotations
import logging
from typing import TYPE_CHECKING, cast

from .info import RoomInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..base.allocator import AllocatorClient
    from ..base.mc import MatchComposerClient
    from ..base.rcss import RcssClient


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
            from ..base.rcss import RcssClient

            self.__rcss = RcssClient(
                self.info.base_url_rcss,
                timeout=self.client.timeout,
            )
        assert self.__rcss is not None
        return self.__rcss

    @property
    def mc(self) -> MatchComposerClient:
        if self.__mc is None:
            from ..base.mc import MatchComposerClient

            self.__mc = MatchComposerClient(
                self.info.base_url_mc,
                timeout=self.client.timeout,
            )
        assert self.__mc is not None
        return self.__mc

    def release(self, *, force: bool = True) -> None:
        try:
            self.rcss.shutdown(force=force)
        except Exception as exc:
            logger.warning("Failed to shutdown room %s via rcss endpoint: %s", self.info.name, exc)
            try:
                self.client.drop_room(self.info.name)
            except Exception as err:
                logger.warning("Failed to drop GameServer for room %s: %s", self.info.name, err)
                raise err from exc

    def heartbeat(self) -> None:
        self.client.heartbeat_room(self.info.name)
