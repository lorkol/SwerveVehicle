
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PathController.Types import Control_Vector
from enum import Enum
from ActuatorController.ActuatorController import ActuatorController
from Types import State2D, State6D


class ControllerTypes(Enum):
    LQR = "LQR"
    MRAC = "MRAC"
    SMC = "SMC"
    MPPI = "MPPI"
    MPC = "MPC"
        
class Controller(ABC):
    def __init__(self, robot_controller: ActuatorController):
        self.actuator: ActuatorController = robot_controller
    
    @abstractmethod
    def get_command(self, state: State6D, debug: bool = False) -> Control_Vector:
        """Compute cost for a given control sequence."""
        pass
    
    @abstractmethod
    def is_stabilized(self, current_state: State6D, pos_tol: float = 0.01, vel_tol: float = 0.0001) -> bool:
        """Return True if the robot is close enough to the goal and nearly stopped."""
        # return (pos_error < pos_tol) and (vel_error < vel_tol) # type: ignore
        pass
    
    @abstractmethod
    def get_reference_state(self, current_pose: State2D) -> State6D:
        """Get the reference state for the controller given the current robot pose."""
        pass    