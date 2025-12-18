from typing import Callable
import numpy as np
from PathController.Controller import Controller
from PathController.Types import State_Vector, Control_Vector, CONTROL_SIZE
from ActuatorController.ActuatorController import ActuatorController
from Types import State2D, State6D


class MRACController(Controller):
    """
    TODO: Class docstring.
    """
    def __init__(self, robot_controller: ActuatorController, get_reference_method: Callable[[np.ndarray], np.ndarray],
                 dt: float = 0.1, alpha_min: float = 0.5, alpha_max: float = 3.0,
                 gamma: float = 0.5, kp: float = 8.0, kv: float = 4.0):
        """
        TODO: Add docstring.
        Args:
            TODO: describe parameters
"""
        super().__init__(robot_controller)        
        self.get_reference_method: Callable[[np.ndarray], np.ndarray] = get_reference_method
        self._dt: float = dt
        self._alpha_min: float = alpha_min
        self._alpha_max: float = alpha_max

        # Reference Model State [x, y, th, vx, vy, vth]
        self.xm: State_Vector = np.zeros(6) 
        self.alpha_hat: np.ndarray = np.array([1.0, 1.0, 1.0]) 
        
        self._gamma: float = gamma   
        self._kp: float = kp      
        self._kv: float = kv
        
        self._initialized: bool = False

    def get_command(self, state: State_Vector, debug: bool = False) -> Control_Vector:
        # --- 1. State Conversion (Robot -> Global) ---
        """
        TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        x, y, theta = state[0], state[1], state[2]
        vx_G, vy_G, v_theta = state[3], state[4], state[5]
        
        c, s = np.cos(theta), np.sin(theta)
        # vx_G = c * vx_R - s * vy_R
        # vy_G = s * vx_R + c * vy_R
        current_global = np.array([x, y, theta, vx_G, vy_G, v_theta])

        # --- 2. Initialization Safety ---
        # On the very first run, snap the internal model to the robot's actual position.
        # Otherwise, the error will be huge (Robot at 10, Model at 0) and gains will explode.
        if not self._initialized:
            self.xm = current_global.copy()
            self._initialized = True

        # --- 3. Get Reference (The Rabbit) ---
        # Get the moving target based on path geometry
        ref: State6D = self.get_reference_method(np.array([x, y, theta]))

        # --- 4. Calculate Tracking Error (Sync Times) ---
        # Compare Robot(t) vs Model(t) BEFORE updating the model
        tracking_err = current_global[:3] - self.xm[:3]

        # --- 5. Update Adaptation (Based on Sync Error) ---
        # If robot is behind model, increase alpha
        self.alpha_hat += -self._gamma * tracking_err * self._dt
        # Clamp to prevent instability (e.g. don't let it reverse control or grow infinite)
        self.alpha_hat = np.clip(self.alpha_hat, self._alpha_min, self._alpha_max)

        # --- 6. Compute Control Signal ---
        # Calculate Nominal Control
        real_pos_err = current_global[:3] - ref[:3]
        real_pos_err[2] = (real_pos_err[2] + np.pi) % (2 * np.pi) - np.pi # Angle wrap
        # real_pos_err[2] *= 0.1  # Reduce yaw error influence
        
        vel_error = current_global[3:6] - ref[3:6]
        u_nominal = -self._kp * real_pos_err - self._kv * vel_error

        # Apply Adaptive Gain
        u_global = u_nominal * self.alpha_hat

        # --- 6.5. Acceleration Limiting ---
        # Clip accelerations to physically achievable limits
        max_lin = self.actuator.max_linear_accel * 0.9  # 90% safety margin
        max_ang = self.actuator.max_angular_accel * 0.9
        
        # Clip linear acceleration magnitude while preserving direction
        lin_accel_mag = np.sqrt(u_global[0]**2 + u_global[1]**2)
        if lin_accel_mag > max_lin:
            scale = max_lin / lin_accel_mag
            u_global[0] *= scale
            u_global[1] *= scale
        
        # Clip angular acceleration
        u_global[2] = np.clip(u_global[2], -max_ang, max_ang)

        # --- 7. Update Reference Model (Prepare for NEXT Step) ---
        # Now move the model to t+1 so it is ready for the next loop.
        # Calculate acceleration for the model based on where it is NOW.
        pos_err_model = self.xm[:3] - ref[:3]
        pos_err_model[2] = (pos_err_model[2] + np.pi) % (2 * np.pi) - np.pi
        
        vel_error_model = self.xm[3:6] - ref[3:6]
        
        acc_model = -self._kp * pos_err_model - self._kv * vel_error_model
        
        # Integrate Model
        self.xm[3:6] += acc_model * self._dt
        self.xm[:3] += self.xm[3:6] * self._dt

        # --- 8. Frame Rotation (Global -> Robot) ---
        # The ActuatorController needs accelerations in the ROBOT frame.
        ax_G, ay_G, alpha_cmd = u_global
        
          
        if debug:
            print(f"[MRAC] Current State (Global): {current_global}")
            print(f"[MRAC] Reference State: {ref}")
            print(f"[MRAC] State Pos Error: {real_pos_err}")
            print(f"[MRAC] State Vel Error: {vel_error}")
            print(f"[MRAC] Acceleration Components: ax_G={ax_G}, ay_G={ay_G}, alpha_cmd={alpha_cmd}")
        
        # Rotation Matrix Transpose (Global to Robot)
        ax_R = c * ax_G + s * ay_G
        ay_R = -s * ax_G + c * ay_G
        
        u_robot = np.array([ax_R, ay_R, alpha_cmd])

        # --- 9. Inverse Dynamics ---
        deltas, torques = self.actuator.get_angles_and_torques(u_robot)

        # --- 10. Pack Output ---
        control_vec = np.zeros(CONTROL_SIZE)
        control_vec[0:4] = torques
        control_vec[4:8] = deltas
        
        return control_vec
    
    def is_stabilized(self, current_state: State6D, pos_tol: float = 0.01, vel_tol: float = 0.0001) -> bool:
            # Check if position error and velocity error are within tolerances
            """
            TODO: Add docstring.

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