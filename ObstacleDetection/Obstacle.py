from __future__ import annotations

from typing import Dict, Optional, List, Union
from Types import ConvexShape, ObstacleType, Point2D


class Obstacle:
    def __init__(self, obstacle_type: ObstacleType = ObstacleType.Static, shape: ConvexShape = ConvexShape.Circle,
                 points: Optional[List[Point2D]] = None, center: Optional[Point2D] = None, radius: Optional[float] = None) -> None:
        self.type: ObstacleType = obstacle_type
        '''The type of the obstacle (static or dynamic).'''
        self.shape: ConvexShape = shape
        '''The shape of the obstacle (circle or polygon).'''
        self.points: List[Point2D] = points if points is not None else []
        '''The vertices of the polygon if the shape is polygon.'''
        self.center: Optional[Point2D] = center
        '''The center of the circle if the shape is circle.'''
        self.radius: Optional[float] = radius
        '''The radius of the circle if the shape is circle.'''
        if self.shape == ConvexShape.Polygon and not self.points:
            raise ValueError("Polygon shape must have points defined.")
        if self.shape == ConvexShape.Circle and (self.center is None or self.radius is None):
            raise ValueError("Circle shape must have center and radius defined.")
        
def load_obstacles(obs_list: List[Dict[str, Union[str, List[Point2D], float]]]) -> List[Obstacle]:
    """Loads obstacles from a list of dictionaries\n
    Only includes static obstacles."""
    obstacles: List[Obstacle] = []
    for obs in obs_list:
        shape = ConvexShape[obs.get("Shape", "Circle")] # type: ignore
        points: List[Point2D] = obs["Points"] if "Points" in obs else [] # type: ignore
        center: Optional[Point2D] = tuple(obs["Center"]) if "Center" in obs else None # type: ignore
        radius: Optional[float] = obs["Radius"] if "Radius" in obs else None # type: ignore
        
        obstacle = Obstacle(
            shape=shape,
            points=[point for point in points] if points else None,
            center=center,
            radius=radius
        )
        obstacles.append(obstacle)
    return obstacles


class MovingObstacle(Obstacle):
    def __init__(self, center: Point2D, radius: float, max_velocity: float, max_acceleration: float) -> None:
        super().__init__(obstacle_type=ObstacleType.Dynamic, shape=ConvexShape.Circle, center=center, radius=radius)
        self.max_velocity: float = max_velocity
        '''The maximum velocity of the moving obstacle.'''
        self.max_acceleration: float = max_acceleration
        '''The maximum acceleration of the moving obstacle.'''
        