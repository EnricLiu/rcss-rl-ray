"""RCSS multi-agent reinforcement learning environment package.

Public API:
- GameServicer / pb2: gRPC service implementation and protobuf message definitions
- AllocatorClient: Room allocator REST client
- RCSSEnv: Ray/RLlib-compatible multi-agent Gymnasium environment
"""

from .grpc_srv import GameServicer, pb2
from client import AllocatorClient

from .env import RCSSEnv
