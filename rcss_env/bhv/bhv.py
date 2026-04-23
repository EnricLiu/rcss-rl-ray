from abc import ABC, abstractmethod
from typing import Optional

from ..grpc_srv.proto import pb2

class BhvMixin(ABC):
    @abstractmethod
    def parse(self, wm: Optional[pb2.WorldModel]) -> pb2.PlayerAction:
        pass