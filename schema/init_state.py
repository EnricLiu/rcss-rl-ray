from ._base import SchemaModel
from .position import Position

class RoomInitState(SchemaModel):
    """Optional initial-state overrides for a simulation room.

    Attributes:
        ball: Normalised initial ball position; None uses the default.
        timestep: Starting simulation timestep, must be in [0, 6000].
    """

    ball: Position | None = None
