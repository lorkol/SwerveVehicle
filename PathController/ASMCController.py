import math
from typing import Callable

import numpy as np
from ActuatorController.ActuatorController import ActuatorController
from PathController.Controller import Control_Vector, Controller
from Types import State2D, State6D

class ACMSController(Controller):
    """
    TODO: Class docstring.    
    """
    def __init__(self, robot_controller: ActuatorController, get_reference_method: Callable[[np.ndarray], np.ndarray], k_d: np.ndarray, etas: np.ndarray, gammas: np.ndarray, lambda_p: np.ndarray,
                 boundary_layer: float = 0.1, dt: float = 0.1, mass_matrix_min: np.ndarray = np.diag([0.1, 0.1, 0.1]), mass_matrix_max: np.ndarray = np.diag([30.0, 30.0, 30.0])):
        # TODO: expand on args
        """
        Args:
            robot_controller (ActuatorController): _description_
            get_reference_method (Callable[[np.ndarray], np.ndarray]): _description_
            k_d (np.ndarray): _description_
            etas (np.ndarray): _description_
            gammas (np.ndarray): _description_
            boundary_layer (float, optional): _description_. Defaults to 0.1.
            dt (float, optional): _description_. Defaults to 0.1.
            mass_matrix_min (np.ndarray, optional): _description_. Defaults to np.diag([0.1, 0.1, 0.1]).
            mass_matrix_max (np.ndarray, optional): _description_. Defaults to np.diag([20.0, 20.0, 20.0]).
        """
        
        super().__init__(robot_controller)
        self.get_reference_method: Callable[[np.ndarray], np.ndarray] = get_reference_method
        
        # Gains
        self._k_d: np.ndarray = np.diag(k_d)
        self._etas: np.ndarray = np.diag(etas)
        self._gammas: np.ndarray = np.diag(gammas)
        self._lambda: np.ndarray = np.diag(lambda_p)
        
        # Model & Timing
        self._boundary_layer: float = boundary_layer
        self._dt: float = dt
        self._mass_matrix_min: np.ndarray = mass_matrix_min
        self._mass_matrix_max: np.ndarray = mass_matrix_max

    def get_command(self, state: State6D, debug: bool = False) -> Control_Vector:
        """
        Calculates control torques based on a single-loop trajectory tracking law.
        """
        ref_state: State6D = self.get_reference_method(state[:3])
        
        # Position Error (e_p)
        e_p: State2D = ref_state[:3] - state[:3]
        e_p[2] = (e_p[2] + math.pi) % (2*math.pi) - math.pi  # Wrap theta
        
        # Velocity Error (e_v)
        e_v: State2D = ref_state[3:6] - state[3:6]

        # s = e_dot + lambda * e  (or e_v + lambda * e_p)
        s: State2D = e_v + self._lambda @ e_p

        # u_eq = lambda * e_v  (assuming ref_accel is 0)
        accel_ref: State2D = self._lambda @ e_v

        if np.any(self._gammas > 0):
            # Update Law: M_dot = Gamma * (accel_ref * s)
            M_update = self._gammas @ (accel_ref * s) * self._dt
            
            # Update the diagonal mass matrix in the actuator
            current_M = self.actuator.get_M_matrix()
            new_M = current_M + np.diag(M_update)
            
            # Enforce limits to maintain stability
            self.actuator.update_M_matrix(np.clip(new_M, self._mass_matrix_min, self._mass_matrix_max))

        M_hat = self.actuator.get_M_matrix()
        
        # Switching Law: u_switch = M_inv * (Kd*s + eta*tanh(s/phi))
        feedback_correction = self._k_d @ s + self._etas @ np.tanh(s / self._boundary_layer)
        
        # Total Command in Global Frame
        accel_cmd_global = accel_ref @ M_hat + feedback_correction

        # --- Transform to Robot Frame ---
        theta: float = state[2]
        c, sn = math.cos(theta), math.sin(theta)
        
        ax_R = c * accel_cmd_global[0] + sn * accel_cmd_global[1]
        ay_R = -sn * accel_cmd_global[0] + c * accel_cmd_global[1]
        alpha_R = -accel_cmd_global[2]
        
        u_robot = np.array([ax_R, ay_R, alpha_R])
        
        u_robot = np.clip(u_robot, -20.0, 20.0)

        steering_angles, wheel_torques = self.actuator.get_angles_and_torques(u_robot)
        
        control_cmd: Control_Vector = np.zeros(9)
        control_cmd[0:4] = wheel_torques
        control_cmd[4:8] = steering_angles
        
        return control_cmd
    
    def is_stabilized(self, current_state: State6D, pos_tol: float = 0.01, vel_tol: float = 0.0001) -> bool:
        """Return True if the robot is close enough to the goal and nearly stopped."""
        pos_error: float = np.linalg.norm(current_state[0:3] - self.get_reference_state(current_state[:3])[0:3]) # type: ignore
        vel_error: float = np.linalg.norm(current_state[3:6]) # type: ignore
        return (pos_error < pos_tol) and (vel_error < vel_tol)
    
    def get_reference_state(self, current_pose: State2D) -> State6D:
        """Get the reference state for the controller given the current robot pose."""
        return self.get_reference_method(current_pose)