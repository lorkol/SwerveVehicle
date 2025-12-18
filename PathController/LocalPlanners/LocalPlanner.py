
from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from Types import State2D, State6D
from ActuatorController.ActuatorController import ActuatorController
import numpy as np

class LocalPlannerTypes(Enum):
    """
    TODO: Class docstring.
    """
    PurePursuit = "PurePursuit"
    
class LocalPlanner(ABC):
    """
    TODO: Class docstring.
    """
    def __init__(self, robot_controller: ActuatorController, path_points: List[State2D]):
        """
        TODO: Add docstring.

        Args:
            TODO: describe parameters
        """
        self.actuator: ActuatorController = robot_controller
        self.path: np.ndarray = np.array(path_points)  # Convert path to numpy array for fast operations
        
    @abstractmethod
    def get_reference_state(self, current_pose: State2D, debug: bool = False) -> State6D:
        """Get the reference state for the controller given the current robot pose."""
        pass

