import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from Scene.Robot import Robot

class ActuatorController:
    def __init__(self, robot: Robot) -> None:
        """Initializes the actuator controller for a swerve drive robot."""
        #Dimensions and physical properties
        self.m: float = robot.mass
        self.I: float = robot.inertia
        self._M_MATRIX: np.ndarray = np.diag([self.m, self.m, self.I])
        self.r: float = robot.wheel_radius
        self.l: float = robot.length
        self.w: float = robot.width
        
        #Wheels/tires rotation/forces
        self.max_wheel_rotation_speed: float = robot.max_wheel_rotation_speed
        self.max_wheel_force: float = robot.max_wheel_torque / self.r
        
        #Steering/delta/phis
        # TODO: Consider how to use the limitations on the steering directions
        self.max_steering_speed: float = robot.max_steering_speed
        self.max_steering_acceleration: float = robot.max_steering_acceleration # TODO: Check if I can use the torque instead
    
    def _create_A_Matrix(self, wheel_angles: np.ndarray) -> np.ndarray:
        """Creates the A matrix for the swerve robot based on wheel angles in radians."""
        
        phi1, phi2, phi3, phi4 = wheel_angles
        
        A: NDArray[np.float64] = np.array([
            [np.cos(phi1), np.sin(phi1), self.l * np.sin(phi1) + self.w * np.cos(phi1)],
            [np.cos(phi2), np.sin(phi2), self.l * np.sin(phi2) - self.w * np.cos(phi2)],
            [np.cos(phi3), np.sin(phi3), -self.l * np.sin(phi3) - self.w * np.cos(phi3)],
            [np.cos(phi4), np.sin(phi4), -self.l * np.sin(phi4) + self.w * np.cos(phi4)]]
        )
        
        return A
        
    def _build_B(self, wheel_angles: np.ndarray) -> np.ndarray:
        """Builds the B matrix for the actuator controller."""
        
        delta1, delta2, delta3, delta4 = wheel_angles

        L_pos, W_pos = self.l, self.w

        # Row 3 (Torque) Calculation: xi*sin(delta) - yi*cos(delta)
        tau_1 = L_pos * math.sin(delta1) - (-W_pos) * math.cos(delta1)  # FR
        tau_2 = L_pos * math.sin(delta2) - W_pos * math.cos(delta2)  # FL
        tau_3 = -L_pos * math.sin(delta3) - W_pos * math.cos(delta3)  # RL
        tau_4 = -L_pos * math.sin(delta4) - (-W_pos) * math.cos(delta4)  # RR

        return np.array([
            [math.cos(delta1), math.cos(delta2), math.cos(delta3), math.cos(delta4)],
            [math.sin(delta1), math.sin(delta2), math.sin(delta3), math.sin(delta4)],
            [tau_1, tau_2, tau_3, tau_4]
        ])
    
    # TODO: Implement input current velocities in x and y, to make use of different wheel angles                
    def get_angles_and_torques(self, accels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical inverse kinematics: Find steering angles and wheel torques
        given desired accelerations in Robot Frame.
        
        Strategy:
        1. Point all wheels in direction of linear acceleration (for efficiency)
        2. Distribute torques asymmetrically to create angular acceleration
        
        This naturally creates:
        - Faster wheels on one side for rotation
        - All wheels pointing same direction for linear motion
        
        Args:
            accels_R: Desired accelerations [ax_R, ay_R, a_theta]
            M_mat: Mass/inertia matrix (3x3)
            l: Half-length of robot
            w: Half-width of robot
            r: Wheel radius
            
        Returns:
            (wheel_angles, wheel_torques): Wheel angles and torques
        """
        
        # Step 1: Calculate desired forces in Robot Frame
        F_R = np.dot(self._M_MATRIX, accels)  # F = M * a
        Fx_R, Fy_R, Tau_theta = F_R
        
        # Step 2: Find the direction all wheels should point for linear motion
        # This minimizes steering and is most efficient
        F_linear = np.sqrt(Fx_R**2 + Fy_R**2)
        
        if F_linear < 1e-6:  # Nearly zero linear force - only rotation
            delta_all = 0.0  # Arbitrary direction (no preferred direction)
        else:
            delta_all = np.arctan2(Fy_R, Fx_R)
        
        # All wheels point the same direction
        wheel_angles = np.array([delta_all, delta_all, delta_all, delta_all])
        
        # Step 3: Build B matrix with this common angle
        B_mat = self._build_B(wheel_angles)
        
        # Step 4: Solve for wheel torques to produce both linear AND angular acceleration
        # We have 3 equations (Fx, Fy, Tau) and 4 unknowns (tau1, tau2, tau3, tau4)
        # Use least squares to find the minimum-norm solution
        # This will automatically make wheels on one side faster for rotation
        
        wheel_forces = np.linalg.lstsq(B_mat, F_R, rcond=None)[0]
        wheel_torques = self.r * wheel_forces
        if np.any(np.abs(wheel_torques) > self.max_wheel_force):
            print("Warning: Wheel torques exceed maximum limits.")
            wheel_torques = np.clip(wheel_torques, -self.max_wheel_force, self.max_wheel_force)
        
        return wheel_angles, wheel_torques

    def get_accels(self, wheel_angles: np.ndarray, wheel_torques: np.ndarray) -> np.ndarray:
        """
        Forward dynamics: Given wheel angles and torques, compute resulting accelerations in Robot Frame.
        
        Args:
            wheel_angles: Steering angles of the wheels [delta1, delta2, delta3, delta4]
            wheel_torques: Torques applied at each wheel [tau1, tau2, tau3, tau4]
            
        Returns:
            accels_R: Resulting accelerations [ax_R, ay_R, a_theta]
        """
        # Build B matrix
        B_mat = self._build_B(wheel_angles)
        
        # Calculate forces at wheels
        F_R = (B_mat / self.r).dot(wheel_torques)
        
        # Calculate accelerations in Robot Frame: a = M^{-1} * F
        accels_R = np.linalg.solve(self._M_MATRIX, F_R)
        
        return accels_R
    
    # def get_accels_in_world(self, state, wheel_angles: np.ndarray, wheel_torques: np.ndarray) -> np.ndarray:
    #     robot_accels = self.get_accels(wheel_angles, wheel_torques)
    #     x, y, theta, vx_R, vy_R, v_theta = state
    #     R_mat = np.array([[math.cos(theta), -math.sin(theta)],
    #                       [math.sin(theta), math.cos(theta)]])
    #     V_R = np.array([vx_R, vy_R])
    #     V_G = R_mat.dot(V_R)
    #     vx_G, vy_G = V_G

    #     # --- 3. Assemble dX/dt vector ---
    #     dXdt = np.array([vx_G, vy_G, v_theta, robot_accels[0], robot_accels[1], robot_accels[2]])

    #     return dXdt
        

    def get_accels_jacobian(self, wheel_angles: np.ndarray, wheel_torques: np.ndarray) -> np.ndarray:
        """
        Jacobian of accelerations with respect to wheel torques.
        
        Returns the partial derivatives of accelerations w.r.t. torques for MPC/SQP.
        
        Math:
            a = M^{-1} * B * (tau / r)
            ∂a/∂tau = M^{-1} * B / r
        
        Args:
            wheel_angles: Steering angles of the wheels [delta1, delta2, delta3, delta4]
            wheel_torques: Torques applied at each wheel [tau1, tau2, tau3, tau4] (used for consistency)
            
        Returns:
            jacobian: (3, 4) matrix where jacobian[i, j] = ∂(accel_i)/∂(torque_j)
            Rows: [∂a_x/∂tau, ∂a_y/∂tau, ∂a_theta/∂tau] (each is 1x4)
            Cols: [∂/∂tau_1, ∂/∂tau_2, ∂/∂tau_3, ∂/∂tau_4]
        """
        # Build B matrix with current wheel angles
        B_mat = self._build_B(wheel_angles)
        
        # Jacobian = M^{-1} * B / r
        # This represents how accelerations change with each torque
        M_inv = np.linalg.inv(self._M_MATRIX)
        jacobian = (M_inv @ B_mat) / self.r
        
        return jacobian

#Usage:
    ## Solve the ODE (Integrate the dynamics over time)
    # solution = odeint(
    #     self.get_state_derivatives,
    #     X0, #np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # State Vector: [x, y, theta, vx_R, vy_R, v_theta]
    #     TIME_POINTS, #TIME_POINTS = np.linspace(0, TIME_TOTAL, TIME_STEPS)
    #     args=(WHEEL_ANGLES, WHEEL_TORQUES)
    # )
    
    def get_state_derivatives(self, X: np.ndarray, wheel_angles: np.ndarray, wheel_torques: np.ndarray) -> np.ndarray:
        """
        The function for the ODE solver (odeint). It calculates the derivatives dX/dt.\n
        X: [x, y, theta, vx_R, vy_R, v_theta]\n
        dX/dt: [vx_G, vy_G, v_theta, ax_R, ay_R, a_theta]
        """
        # Unpack current state
        x, y, theta, vx_R, vy_R, v_theta = X

        # --- 1. Calculate accelerations in the Robot Frame ---
        B_mat = self._build_B(wheel_angles)
        F_R = (B_mat / self.r).dot(wheel_torques)

        # Calculate accelerations in the Robot Frame: a = M^{-1} * F
        accels_R = np.linalg.solve(self._M_MATRIX, F_R)
        ax_R, ay_R, a_theta = accels_R

        # --- 2. Calculate velocities in the Global Frame ---
        R_mat = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ])
        V_R = np.array([vx_R, vy_R])
        V_G = R_mat.dot(V_R)
        vx_G, vy_G = V_G

        # --- 3. Assemble dX/dt vector ---
        dXdt = np.array([
            vx_G,
            vy_G,
            v_theta,
            ax_R,
            ay_R,
            a_theta
        ])

        return dXdt
