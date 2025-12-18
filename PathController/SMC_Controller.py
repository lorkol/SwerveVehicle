from typing import Callable
import numpy as np
from PathController.Controller import Controller
from PathController.Types import State_Vector, Control_Vector, CONTROL_SIZE
from ActuatorController.ActuatorController import ActuatorController

class SMCController(Controller):
    """
    TODO: Class docstring.

    Attributes:
        TODO: describe attributes
    """
    def __init__(self, robot_controller: ActuatorController, get_reference_method: Callable[[np.ndarray], np.ndarray], lambda_gains: np.ndarray, k_gains: np.ndarray, boundary_layer: float = 0.1):
        """
        TODO: Add docstring.

        Args:
            TODO: describe parameters
    """
        self.actuator: ActuatorController = robot_controller
        self.get_reference_method: Callable[[np.ndarray], np.ndarray] = get_reference_method
        
        # --- SMC Tuning Parameters ---
        
        # Lambda (λ): The slope of the sliding surface. 
        # Determines how fast the error decays once we are "on the rails" (on the surface).
        # Higher = Faster decay, but requires more control authority.
        self._lambda_gain: np.ndarray = np.diag(lambda_gains) 


        # K (Gain): The switching gain (Aggressiveness).
        # Determines how hard we push to get back to the surface if we are off.
        # Must be larger than the upper bound of your system's disturbances (friction/model error).
        self._k_gain: np.ndarray = np.diag(k_gains)
        
        # Phi (Φ): Boundary layer width.
        # Smooths the transition around s=0 to prevent chattering.
        # Tanh(s/phi) behaves like sign(s) but is smooth near zero.
        self._boundary_layer: float = boundary_layer

    def get_command(self, state: State_Vector, dt: float = 0.1) -> Control_Vector:
        # --- 1. State Conversion (Robot -> Global) ---
        """
        TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        x, y, theta = state[0], state[1], state[2]
        vx_R, vy_R, v_theta = state[3], state[4], state[5]
        
        c, s_ang = np.cos(theta), np.sin(theta)
        vx_G: float = c * vx_R - s_ang * vy_R
        vy_G: float = s_ang * vx_R + c * vy_R
        
        current_global_pos = np.array([x, y, theta])
        current_global_vel = np.array([vx_G, vy_G, v_theta])

        # --- 2. Get Reference (The "Carrot") ---
        # ref_state: [x_r, y_r, th_r, vx_r, vy_r, vth_r]
        ref_state: np.ndarray = self.get_reference_method(current_global_pos)
        
        ref_pos = ref_state[:3]
        ref_vel = ref_state[3:6]
        
        # For this implementation, we assume constant speed segments (ref_accel = 0)
        # If your path generator supported curvature acceleration, you would add it here.
        ref_accel = np.zeros(3) 

        # --- 3. Calculate Errors ---
        # e = x_ref - x
        e_pos = ref_pos - current_global_pos
        
        # Angle Wrapping for Theta Error: Ensure range [-pi, pi]
        e_pos[2] = (e_pos[2] + np.pi) % (2 * np.pi) - np.pi
        
        # e_dot = v_ref - v
        e_vel = ref_vel - current_global_vel

        # --- 4. Define Sliding Surface (s) ---
        # The goal is to drive 's' to zero.
        # s = e_dot + lambda * e
        s = e_vel + self._lambda_gain @ e_pos

        # --- 5. Compute Control Law (u) ---
        # We derived: u = a_ref + lambda * e_dot + K * tanh(s/phi)
        
        # Term A: Equivalent Control (The effort to stay ON the surface if we are already there)
        # u_eq = a_ref + lambda * e_dot
        u_eq = ref_accel + self._lambda_gain @ e_vel
        
        # Term B: Switching Control (The robust effort to force us TO the surface)
        # u_switch = K * tanh(s / phi)
        # Note: We use tanh for smoothness. Standard SMC uses np.sign(s).
        u_switch = self._k_gain @ np.tanh(s / self._boundary_layer)
        
        # Total Global Acceleration Command
        u_global = u_eq + u_switch

        # --- 6. Frame Rotation (Global -> Robot) ---
        # The actuator controller requires acceleration in the Robot Frame.
        ax_G, ay_G, alpha_cmd = u_global
        
        # Rotate linear acceleration vector by -theta
        ax_R = c * ax_G + s_ang * ay_G
        ay_R = -s_ang * ax_G + c * ay_G
        
        u_robot = np.array([ax_R, ay_R, alpha_cmd])

        # --- 7. Inverse Dynamics ---
        deltas, torques = self.actuator.get_angles_and_torques(u_robot)

        # --- 8. Pack Output ---
        control_vec = np.zeros(CONTROL_SIZE)
        control_vec[0:4] = torques
        control_vec[4:8] = deltas
        
        return control_vec
    
    def get_reference_state(self, current_pose: State_Vector) -> State_Vector:
        """
        TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        return self.get_reference_method(current_pose)
    
    def is_stabilized(self, current_state: State_Vector, pos_tol: float = 0.01, vel_tol: float = 0.0001) -> bool:
        # Check if position error and velocity error are within tolerances
        """
        TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        pos_error: float = np.linalg.norm(current_state[0:3] - self.get_reference_method(current_state[:3])[0:3]) # type: ignore
        vel_error: float = np.linalg.norm(current_state[3:6]) # type: ignore
        return (pos_error < pos_tol) and (vel_error < vel_tol)