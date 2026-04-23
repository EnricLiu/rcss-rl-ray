from typing import override, Optional

from ..grpc_srv.proto import pb2
from .bhv import BhvMixin

class BhvNaiveNeck(BhvMixin):
    def __init__(self, count_threshold: int = 3) -> None:
        self.count_threshold = count_threshold

    @override
    def parse(self, wm: Optional[pb2.WorldModel]) -> pb2.PlayerAction:
        neck_turn_to_ball_or_scan = pb2.Neck_TurnToBallOrScan(
            count_threshold = self.count_threshold,
        )
        return pb2.PlayerAction(
            neck_turn_to_ball_or_scan=neck_turn_to_ball_or_scan
        )