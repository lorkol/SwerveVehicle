import numpy as np
import scipy.linalg
from PathController.Controller import Controller
from ActuatorController.ActuatorController import ActuatorController
from PathController.PathReference import ProjectedPathFollower
from PathController.Types import State_Vector, Control_Vector, CONTROL_SIZE
from Types import Point2D
from typing import List

class LQRController(Controller):
    def __init__(self, robot_controller: ActuatorController, path_points: list[Point2D], Q: List[float], R: List[float], lookahead: float = 0.3, v_desired: float = 1.0, dt: float = 0.1):
        self.actuator: ActuatorController = robot_controller
        self.path_follower: ProjectedPathFollower = ProjectedPathFollower(path_points)
        
        # Tuning Parameters # TODO: from parameters
        self._lookahead: float = lookahead  # meters
        self._v_desired: float = v_desired  # m/s
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

    def get_command(self, state: State_Vector) -> Control_Vector:
        # --- 1. State Conversion (Robot -> Global) ---
        x, y, theta = state[0], state[1], state[2]
        vx_R, vy_R, v_theta = state[3], state[4], state[5]

        # Rotate velocities to global frame for LQR
        c, s = np.cos(theta), np.sin(theta)
        vx_G: float = c * vx_R - s * vy_R
        vy_G: float = s * vx_R + c * vy_R
        
        current_state_global: State_Vector = np.array([x, y, theta, vx_G, vy_G, v_theta])

        # --- 2. Reference Generation ---
        ref_state = self.path_follower.get_reference_state(np.array([x, y]), self._lookahead, self._v_desired)

        # --- 3. LQR Calculation (Global Frame) ---
        error = current_state_global - ref_state
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi # Angle wrap

        # u_global = [ax_G, ay_G, alpha_accel]
        u_global = -self.K @ error

        # --- 4. Frame Rotation (Global -> Robot) ---
        # The ActuatorController needs accelerations in the ROBOT frame.
        # We rotate the linear acceleration vector by -theta.
        ax_G, ay_G, alpha_cmd = u_global
        
        # Rotation Matrix Transpose (Global to Robot)
        # [ c  s ]
        # [ -s c ]
        ax_R = c * ax_G + s * ay_G
        ay_R = -s * ax_G + c * ay_G
        
        u_robot = np.array([ax_R, ay_R, alpha_cmd])

        # --- 5. Inverse Dynamics ---
        # Returns (wheel_angles, wheel_torques)
        deltas, torques = self.actuator.get_angles_and_torques(u_robot)

        # --- 6. Pack Output ---
        control_vec = np.zeros(CONTROL_SIZE)
        control_vec[0:4] = torques # Tau 1-4
        control_vec[4:8] = deltas  # Delta 1-4
        
        return control_vec

