from __future__ import annotations
from typing import List, Optional
from Types import Point2D, ConvexShape

from Obstacle import Obstacle
from Types import ObstacleType


class MovingObstacle(Obstacle):
    def __init__(self, center: Point2D, radius: float, max_velocity: float, max_acceleration: float) -> None:
        super().__init__(obstacle_type=ObstacleType.Dynamic, shape=ConvexShape.Circle, center=center, radius=radius)
        self.max_velocity: float = max_velocity
        '''The maximum velocity of the moving obstacle.'''
        self.max_acceleration: float = max_acceleration
        '''The maximum acceleration of the moving obstacle.'''
        