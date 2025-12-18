from typing import Any, List, Dict
from ObstacleDetection.Obstacle import Obstacle, load_obstacles

class Map:
    """
    TODO: Class docstring.

    Attributes:
        TODO: describe attributes
    """
    def __init__(self, map_object: Dict[str, Any]) -> None:
        self.length: float = map_object["Dimensions"]["Length"]
        self.width: float = map_object["Dimensions"]["Width"]
        self.friction_coefficient: float = map_object["FrictionCoefficient"]
        self.obstacles: List[Obstacle] = load_obstacles(map_object["Obstacles"])