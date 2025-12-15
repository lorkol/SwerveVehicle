
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeAlias


Point2D: TypeAlias = Tuple[float, float]

State2D: TypeAlias = Tuple[float, float, float]  # (x, y, theta)
'''Represents (x, y, theta) the state of the vehicle in 2D space with orientation.'''

PathType: TypeAlias = List[State2D]
'''A path represented as a list of 2D states (x, y, theta).'''

OptionalPathType: TypeAlias = Optional[List[State2D]]
'''An optional path which can be None.'''

ConfigDict: TypeAlias = Dict[str, Any]
"""A dictionary for configuration parameters."""

class ConvexShape(Enum):
    """Defines the shape of the obstacle."""
    Circle = auto()
    Polygon = auto()

class ObstacleType(Enum):
    """Defines if the obstacle is static or dynamic."""
    Static = auto()
    Dynamic = auto()