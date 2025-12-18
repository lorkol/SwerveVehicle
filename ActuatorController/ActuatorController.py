import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from Scene.Robot import Robot

class ActuatorController:
    """Actuator Controller for a Swerve Drive Robot.
    This class computes the necessary wheel steering angles and torques
    to achieve desired accelerations in the robot frame.
    Also does the inverse: given wheel angles and torques, compute resulting accelerations.
    """
    def __init__(self, robot: Robot) -> None:
        """Initializes the actuator controller for a swerve drive robot."""
        #Dimensions and physical properties
        self._m: float = robot.mass
        self._I: float = robot.inertia
        self._M_MATRIX: np.ndarray = np.diag([self._m, self._m, self._I])
        self._r: float = robot.wheel_radius
        self._l: float = robot.length
        self._w: float = robot.width
        
        #Wheels/tires rotation/forces
        self._max_wheel_rotation_speed: float = robot.max_wheel_rotation_speed
        self._max_wheel_force: float = robot.max_wheel_torque / self._r
        
        #Steering/delta/phis
        # TODO: Consider how to use the limitations on the steering directions
        self._max_steering_speed: float = robot.max_steering_speed
        self._max_steering_acceleration: float = robot.max_steering_acceleration # TODO: Check if I can use the torque instead
        
        # Compute maximum achievable accelerations from physical limits
        # Max linear acceleration: all 4 wheels pushing in same direction
        # a_max = (4 * F_max) / m = (4 * tau_max / r) / m
        self._max_linear_accel: float = (4.0 * self._max_wheel_force) / self._m
        
        # Max angular acceleration: wheels creating pure torque
        # For a rectangular chassis, max torque arm ≈ sqrt(l² + w²)
        # alpha_max = (4 * F_max * arm) / I
        arm_length = math.sqrt(self._l**2 + self._w**2)
        self._max_angular_accel: float = (4.0 * self._max_wheel_force * arm_length) / self._I
    
    @property
    def max_linear_accel(self) -> float:
        """Maximum achievable linear acceleration (m/s²) based on wheel torque limits."""
        return self._max_linear_accel
    
    @property
    def max_angular_accel(self) -> float:
        """Maximum achievable angular acceleration (rad/s²) based on wheel torque limits."""
        return self._max_angular_accel
    
    def _create_A_Matrix(self, wheel_angles: np.ndarray) -> np.ndarray:
        """Creates the A matrix for the swerve robot based on wheel angles in radians."""
        
        phi1, phi2, phi3, phi4 = wheel_angles
        
        A: NDArray[np.float64] = np.array([
            [np.cos(phi1), np.sin(phi1), self._l * np.sin(phi1) + self._w * np.cos(phi1)],
            [np.cos(phi2), np.sin(phi2), self._l * np.sin(phi2) - self._w * np.cos(phi2)],
            [np.cos(phi3), np.sin(phi3), -self._l * np.sin(phi3) - self._w * np.cos(phi3)],
            [np.cos(phi4), np.sin(phi4), -self._l * np.sin(phi4) + self._w * np.cos(phi4)]]
        )
        
        return A
        
    def _build_B(self, wheel_angles: np.ndarray) -> np.ndarray:
        """Builds the B matrix for the actuator controller."""
        
        delta1, delta2, delta3, delta4 = wheel_angles

        L_pos, W_pos = self._l, self._w

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
    
    def get_angles_and_torques(self, accels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical inverse dynamics: Find steering angles and wheel torques
        given desired accelerations in Robot Frame.
        
        Strategy: Vector Addition (Linear Force + Rotational Force)
        """
        # Step 1: Calculate desired Net Forces/Moments at CoM
        F_R = np.dot(self._M_MATRIX, accels)  # F = M * a
        Fx_target, Fy_target, Tau_target = F_R
        
        # Step 2: Distribute forces to each wheel
        # We assume equal load distribution (1/4 force per wheel)
        # Wheel positions relative to center (Order: FR, FL, RL, RR based on your B matrix)
        # Based on build_B: 
        # 1: (+L, -W), 2: (+L, +W), 3: (-L, +W), 4: (-L, -W)
        wheel_positions = np.array([
            [self._l, -self._w], # 1
            [self._l, self._w],  # 2
            [-self._l, self._w], # 3
            [-self._l, -self._w] # 4
        ])
        
        # Radius squared for torque calculations (r^2 = x^2 + y^2)
        # All wheels are equidistant in a rectangular chassis
        r_sq = self._l**2 + self._w**2
        
        wheel_angles = []
        wheel_torques = []
        
        for i in range(4):
            rx, ry = wheel_positions[i]
            
            # A. Linear Component (Pure Translation)
            # Distribute total force equally
            fx_lin = Fx_target / 4.0
            fy_lin = Fy_target / 4.0
            
            # B. Rotational Component (Pure Rotation)
            # Torque = r x F. To create positive Tau (CCW), force must be tangent CCW.
            # Tangent vector at (x,y) is (-y, x)
            # Force magnitude F_rot = Tau / (4 * r)
            # F_rot_vec = (Tau / 4r) * (-y/r, x/r) = (Tau / 4r^2) * (-y, x)
            
            common_factor = Tau_target / (4.0 * r_sq)
            fx_rot = common_factor * (-ry)
            fy_rot = common_factor * (rx)
            
            # C. Total Force Vector at Wheel
            fx_total = fx_lin + fx_rot
            fy_total = fy_lin + fy_rot
            
            # D. Convert to Steering Angle and Torque
            # Steering angle is direction of force
            delta = math.atan2(fy_total, fx_total)
            
            # Force magnitude
            f_mag = math.sqrt(fx_total**2 + fy_total**2)
            
            # Direction Check:
            # Wheels can spin forward or backward. Optimization:
            # If the required angle is > 90 deg from current, flip force and angle?
            # For this simple implementation, we assume infinite steering speed 
            # and just output the exact vector direction.
            
            torque = f_mag * self._r
            
            # Optimization: If the force opposes the steering direction, flip torque?
            # The atan2 above ensures force is always positive in the direction of delta.
            # However, if we wanted to support reversing the wheel to minimize steering:
            # (Not implemented here to keep it stable for now)
            
            wheel_angles.append(delta)
            wheel_torques.append(torque)
            
        wheel_angles = np.array(wheel_angles)
        wheel_torques = np.array(wheel_torques)

        # Clip torques
        if np.any(np.abs(wheel_torques) > self._max_wheel_force):
            # Scale down to maintain direction, or just clip?
            # Clipping allows some wheels to work while others saturate (better for control)
            wheel_torques = np.clip(wheel_torques, -self._max_wheel_force, self._max_wheel_force)
        
        # Sanity Check: Verify forward dynamics recovers the input accelerations
        recovered_accels = self.get_accels(wheel_angles, wheel_torques)
        error = np.linalg.norm(recovered_accels - accels)
        if error > 0.1:
            print(f"WARNING: Dynamics mismatch! Input: {accels}, Recovered: {recovered_accels}, Error: {error:.4f}")
            
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
        F_R = (B_mat / self._r).dot(wheel_torques)
        
        # Calculate accelerations in Robot Frame: a = M^{-1} * F
        accels_R = np.linalg.solve(self._M_MATRIX, F_R)
        
        return accels_R
    
    def get_accels_in_world(self, state: np.ndarray, wheel_angles: np.ndarray, wheel_torques: np.ndarray) -> np.ndarray:
        """
        Given the robot state, wheel angles, and wheel torques, compute the linear accelerations in the World Frame.
        
        Converts robot frame accelerations to global frame using rotation matrix.
        
        Args:
            state: Current state [x, y, theta, vx_R, vy_R, v_theta, d1, d2, d3, d4]
            wheel_angles: Steering angles of the wheels [delta1, delta2, delta3, delta4] in robot frame
            wheel_torques: Torques applied at each wheel [tau1, tau2, tau3, tau4]
            
        Returns:
            a_G: Accelerations in global frame [ax_G, ay_G, a_theta]
        """
        # Wheel angles are measured in ROBOT frame (relative to robot body)
        # They are NOT affected by robot's global orientation theta
        robot_accels = self.get_accels(wheel_angles, wheel_torques)
        x, y, theta, vx_R, vy_R, v_theta = state[:6]
        ax_R, ay_R, a_theta = robot_accels
        
        # Rotation matrix from robot frame to global frame
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Convert linear accelerations: a_G = R * a_R
        # This is the straightforward transformation without centripetal terms
        ax_G = cos_theta * ax_R - sin_theta * ay_R
        ay_G = sin_theta * ax_R + cos_theta * ay_R
        
        return np.array([ax_G, ay_G, a_theta])
    

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
        jacobian = (M_inv @ B_mat) / self._r
        
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
        F_R = (B_mat / self._r).dot(wheel_torques)

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
