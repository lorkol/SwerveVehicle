
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Optional
from Types import State2D, State6D, NP3DPoint
from ActuatorController.ActuatorController import ActuatorController
import numpy as np

class LocalPlannerTypes(Enum):
    PurePursuit = "PurePursuit"
    
class LocalPlanner(ABC):
    def __init__(self, robot_controller: ActuatorController, path_points: List[State2D]):
        self.actuator: ActuatorController = robot_controller
        self.path: np.ndarray = np.array(path_points)  # Convert path to numpy array for fast operations
        
    @abstractmethod
    def get_reference_state(self, current_pose: NP3DPoint, debug: bool = False) -> State6D:
        """Get the reference state for the controller given the current robot pose."""
        pass

