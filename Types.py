
from enum import Enum, auto
from typing import TypeAlias


Point2D: TypeAlias = tuple[float, float]

class ConvexShape(Enum):
    """Defines the shape of the obstacle."""
    Circle = auto()
    Polygon = auto()

class ObstacleType(Enum):
    """Defines if the obstacle is static or dynamic."""
    Static = auto()
    Dynamic = auto()