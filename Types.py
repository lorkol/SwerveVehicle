
from enum import Enum, auto
from typing import Tuple, TypeAlias


Point2D: TypeAlias = Tuple[float, float]

State2D: TypeAlias = Tuple[float, float, float]  # (x, y, theta)
'''Represents (x, y, theta) the state of the vehicle in 2D space with orientation.'''

class ConvexShape(Enum):
    """Defines the shape of the obstacle."""
    Circle = auto()
    Polygon = auto()

class ObstacleType(Enum):
    """Defines if the obstacle is static or dynamic."""
    Static = auto()
    Dynamic = auto()