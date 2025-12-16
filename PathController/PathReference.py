from abc import ABC, abstractmethod
import numpy as np


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
    
    # TODO: Generalize so it can either get time or current_state
    def get_reference_state(self, t: float) -> np.ndarray:
        return np.array([20.0, 20., 0.0])
    
    def get_reference_velocity(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def get_reference_acceleration(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])
