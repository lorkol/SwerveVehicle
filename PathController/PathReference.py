from abc import ABC, abstractmethod
import numpy as np

from ObstacleDetection.ObstacleDetector import ObstacleChecker, StaticObstacleChecker

class ReferenceGenerator(ABC):
    @abstractmethod
    def get_reference_state(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_reference_velocity(self) -> np.ndarray:
        pass
    
class SimpleReferenceGenerator(ReferenceGenerator):
    def __init__(self):
        pass
    
    def get_reference_state(self, t: float) -> np.ndarray:
        return np.array([20.0, 20., 0.0])
    
    def get_reference_velocity(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def get_reference_acceleration(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])
