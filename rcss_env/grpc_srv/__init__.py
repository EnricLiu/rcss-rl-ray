"""gRPC service sub-package for communicating with SoccerSimulationProxy sidecars."""

from .proto import pb2
from .servicer import GameServicer, serve
from .batch_queue import BatchQueue
