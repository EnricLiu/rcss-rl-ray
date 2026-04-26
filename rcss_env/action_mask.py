from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from schema.player import PlayerSchema

from .action import Action
from .grpc_srv.proto import pb2

logger = logging.getLogger(__name__)

_MASK_PLACEHOLDER_WARNING_EMITTED = False

class ActionMaskResolver:
    OBSERVATION_KEY = "action_mask"

    def __init__(self, player: PlayerSchema):
        self.__schema = player

    @property
    def schema(self) -> PlayerSchema:
        return self.__schema

    def __resolve_basic(self) -> set[str]: # blocked
        blocked = set()

        if not self.schema.goalie:
            blocked.add(Action.CATCH)

        return blocked

    # TODO
    def __resolve(self, wm: Optional[pb2.WorldModel]) -> set[str]: # blocked
        blocked = self.__resolve_basic()
        if wm is None: return blocked

        if not wm.self.is_kickable:
            blocked.add(Action.KICK)

        return blocked

    def resolve(self, wm: Optional[pb2.WorldModel]) -> np.ndarray:
        blocked = self.__resolve(wm)
        return Action.mask_from_blocked(blocked)

# def _disabled_action_names_from_blocklist(blocklist: PlayerActionList | None) -> set[str]:
#     if blocklist is None:
#         return set()
#
#     disabled: set[str] = set()
#     blocklist_dump = blocklist.model_dump(by_alias=True)
#     for action_name, is_blocked in blocklist_dump.items():
#         if is_blocked:
#             disabled.add(action_name)
#     return disabled
