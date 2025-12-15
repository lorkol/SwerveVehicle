
from abc import ABC, abstractmethod
from PathController.Types import State_Vector, Control_Vector
from enum import Enum
import numpy as np
from ActuatorController.ActuatorController import ActuatorController
from PathController.PathReference import ProjectedPathFollower


class ControllerTypes(Enum):
    LQR = "LQR"
    MRAC = "MRAC"
    SMC = "SMC"
    MPPI = "MPPI"
    MPC = "MPC"
    
    
class Controller(ABC):
    
    def __init__(self, robot_controller: ActuatorController, path_follower: ProjectedPathFollower):
        self.actuator: ActuatorController = robot_controller
        self.path_follower: ProjectedPathFollower = path_follower
    
    @abstractmethod
    def get_command(self, state: State_Vector, debug: bool = False) -> Control_Vector:
        """Compute cost for a given control sequence."""
        pass
    
    def is_stabilized(self, current_state: State_Vector, pos_tol: float = 0.01, vel_tol: float = 0.0001) -> bool:
        """Return True if the robot is close enough to the goal and nearly stopped."""
        pos_error = np.linalg.norm(current_state[0:3] - self.path_follower.path[-1][0:3])
        vel_error = np.linalg.norm(current_state[3:6])
        
        return (pos_error < pos_tol) and (vel_error < vel_tol) # type: ignore