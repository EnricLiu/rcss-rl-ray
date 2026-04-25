import numpy as np
from pydantic import BaseModel

from schema import Position


class Region(BaseModel):
    pos_a: Position
    pos_b: Position

    @staticmethod
    def from_range(x: tuple[float, float], y: tuple[float, float]) -> "Region":
        """Create a Region from x and y ranges."""
        return Region(
            pos_a=Position(x=x[0], y=y[0]),
            pos_b=Position(x=x[1], y=y[1]),
        )

    def dist(self) -> Position:
        return Position(
            x=abs(self.pos_a.x - self.pos_b.x),
            y=abs(self.pos_a.y - self.pos_b.y),
        )

    def sample_gaussian(self, std: float) -> Position:
        x_min, x_max = sorted((self.pos_a.x, self.pos_b.x))
        y_min, y_max = sorted((self.pos_a.y, self.pos_b.y))

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2

        x_sample = np.random.normal(loc=x_center, scale=(x_max - x_min) * std)
        y_sample = np.random.normal(loc=y_center, scale=(y_max - y_min) * std)

        return Position(
            x=float(np.clip(x_sample, x_min, x_max)),
            y=float(np.clip(y_sample, y_min, y_max)),
        )

    def sample_p99(self) -> Position:
        """Sample with a Gaussian whose region bounds correspond to ±3σ."""
        return self.sample_gaussian(std=1 / 6)