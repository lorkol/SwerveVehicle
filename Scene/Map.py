from typing import List
import JsonManager
from ObstacleDetection.Obstacle import Obstacle, load_obstacles

class Map:
    def __init__(self, map_file: str) -> None:
        self._map_json = JsonManager.JsonManager(map_file)
        self.length: float = self._map_json.read_param_value("Dimensions/Length")
        self.width: float = self._map_json.read_param_value("Dimensions/Width")
        self.friction_coefficient: float = self._map_json.read_param_value("FrictionCoefficient")
        self.obstacles: List[Obstacle] = load_obstacles(self._map_json.read_param_value("Obstacles"))