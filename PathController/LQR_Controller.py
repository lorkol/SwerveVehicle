import numpy as np
import scipy.linalg
from PathController.Controller import Controller
from ActuatorController.ActuatorController import ActuatorController
from PathController.Types import Control_Vector, CONTROL_SIZE
from PathController.Controller import Controller
from typing import Callable, List

from Types import State2D, State6D

class LQRController(Controller):
    """TODO: Class docstring.

    Attributes:
        TODO: describe attributes
    """
    def __init__(self, robot_controller: ActuatorController, get_reference_method: Callable[[np.ndarray], np.ndarray], Q: List[float], R: List[float], dt: float = 0.1):
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        super().__init__(robot_controller)
        self.get_reference_method: Callable[[np.ndarray], np.ndarray] = get_reference_method
        self._dt: float = dt

        # 1. Linearized Model (Double Integrator in Global Frame)
        # State: [x, y, theta, vx_G, vy_G, v_theta]
        # Input: [ax_G, ay_G, alpha]
        self.A = np.zeros((6, 6))
        self.A[0, 3] = 1; self.A[1, 4] = 1; self.A[2, 5] = 1 

        self.B = np.zeros((6, 3))
        self.B[3, 0] = 1; self.B[4, 1] = 1; self.B[5, 2] = 1 

        # 2. Costs
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        # 3. Solve Riccati
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P
        print(f"[LQR] Gain Matrix K:\n{self.K}")

    def get_command(self, state: State6D, debug: bool = False) -> Control_Vector:
        # --- 1. State Conversion (Robot -> Global) ---
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        x, y, theta = state[0], state[1], state[2]
        vx_G, vy_G, v_theta = state[3], state[4], state[5]

        # Rotate velocities to global frame for LQR
        c, s = np.cos(theta), np.sin(theta)
        # vx_G: float = c * vx_R - s * vy_R
        # vy_G: float = s * vx_R + c * vy_R
        
        current_state_global: State6D = np.array([x, y, theta, vx_G, vy_G, v_theta])

        # --- 2. Reference Generation ---
        carrot_state: State6D = self.get_reference_method(state[:3])
        
        # --- 3. Construct LQR Reference (The "Masking" Step) ---
        ref_state: State6D = np.zeros(6)

        # POSITION (Masked): Set ref to current pos so error = 0.
        # This prevents the "Waiting for Bus" bug. LQR ignores position error.
        ref_state[0] = x
        ref_state[1] = y
        
        # HEADING: Keep Planner Heading.
        # LQR will fight to keep robot aligned with the path tangent.
        ref_state[2] = carrot_state[2] 
        
        # VELOCITY: Use the Pure Pursuit Vector.
        # LQR will fight to match this velocity vector.
        ref_state[3] = carrot_state[3]
        ref_state[4] = carrot_state[4]
        
        # ANGULAR RATE: Usually 0 (Stop spinning when aligned)
        ref_state[5] = 0.0 

        # --- 4. LQR Error Calculation ---
        error: State6D = current_state_global - ref_state
        
        # Critical: Wrap Heading Error to [-pi, pi]
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi 

        # --- 5. Compute Control Law ---
        # u_global = [ax_G, ay_G, alpha_cmd]
        # Because error[0] and error[1] are 0, this acts as:
        # u = -K_theta*theta_err - K_v*(v_current - v_des)
        u_global = -self.K @ error
       
        # --- 6. Saturation (Physical Limits) ---
        max_lin: float = self.actuator.max_linear_accel * 0.9
        max_ang: float = self.actuator.max_angular_accel * 0.9
        
        # Clip linear acceleration magnitude
        lin_accel_mag: float = np.sqrt(u_global[0]**2 + u_global[1]**2)
        if lin_accel_mag > max_lin:
            scale: float = max_lin / lin_accel_mag
            u_global[0] *= scale
            u_global[1] *= scale
        
        # Clip angular acceleration
        u_global[2] = np.clip(u_global[2], -max_ang, max_ang)

        # --- 7. Rotate Command to Robot Frame ---
        # Actuator Controller expects accelerations relative to the body
        ax_G, ay_G, alpha_cmd = u_global
        
        ax_R: float = c * ax_G + s * ay_G
        ay_R: float = -s * ax_G + c * ay_G
        
        u_robot: np.ndarray = np.array([ax_R, ay_R, alpha_cmd])

        # --- 8. Inverse Dynamics ---
        deltas, torques = self.actuator.get_angles_and_torques(u_robot)

        # --- 9. Output ---
        control_vec: Control_Vector = np.zeros(CONTROL_SIZE)
        control_vec[0:4] = torques
        control_vec[4:8] = deltas
        
        if debug:
            print(f"[LQR] Robot State (Global): {current_state_global}")
            print(f"[LQR] Carrot State: {carrot_state}")
            print(f"[LQR] Reference State: {ref_state}")
            print(f"[LQR] State Error: {error}")
            print(f"[LQR] Control Output (u_global): {u_global}")
            print(f"[LQR] Control Output (Robot Frame): {u_robot}")
            print(f"[LQR] Torques: {torques}, Deltas: {deltas}")
        
        return control_vec

    def is_stabilized(self, current_state: State6D, pos_tol: float = 0.01, vel_tol: float = 0.0001) -> bool:
        # Check if position error and velocity error are within tolerances
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        pos_error: float = np.linalg.norm(current_state[0:3] - self.get_reference_state(current_state[:3])[0:3]) # type: ignore
        vel_error: float = np.linalg.norm(current_state[3:6]) # type: ignore
        return (pos_error < pos_tol) and (vel_error < vel_tol)
    
    
    def get_reference_state(self, current_pose: State2D) -> State6D:
        """Get the reference state for the controller given the current robot pose."""
        return self.get_reference_method(current_pose)