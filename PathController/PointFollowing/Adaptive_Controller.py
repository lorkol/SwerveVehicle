import math
from typing import Tuple, TypeAlias
from ActuatorController.ActuatorController import ActuatorController

import numpy as np
from PathController.LocalPlanners.PathReference import SimpleReferenceGenerator
from PathController.Robot_Sim import Robot_Sim
from Types import State6D

State: TypeAlias = np.ndarray #(3, )
'''np (3,) x, y, theta'''

from scipy.linalg import solve_continuous_lyapunov

class AdaptiveController:
    """TODO: Class docstring.

    Attributes:
        TODO: describe attributes
    """
    def __init__(self, dt: float):
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        self.dt = dt
        
        # --- 1. Gains ---
        # TODO: Get from Params
        self.Kp = np.array([5.0, 5.0, 3.0])  # x, y, theta
        self.Kd = np.array([2.0, 2.0, 1.5])  # vx, vy, omega
        
        # How fast we learn
        # TODO: Get from Params
        self.Gamma = np.array([0.5, 0.1]) # [Mass_rate, Inertia_rate]
        
        # [Estimated Mass, Estimated Inertia]
        # TODO: Get from initial config
        self.theta_hat = np.array([5.0, 0.5]) 
        
        # Limits to keep physics realistic
        self.theta_min = np.array([0.1, 0.01])
        self.theta_max = np.array([50.0, 10.0])

        # We solve A^T P + PA = -Q for each DOF to find the correct error mixing
        # A = [[0, 1], [-kp, -kd]]
        self.P_matrices = []
        for i in range(3): # For x, y, theta
            A_c = np.array([[0, 1], 
                           [-self.Kp[i], -self.Kd[i]]])
            Q_c = np.eye(2) # Symmetric positive definite matrix
            P_c = solve_continuous_lyapunov(A_c.T, -Q_c)
            self.P_matrices.append(P_c)

    def update(self, current_state: State6D, target_state: State6D, target_vel, target_accel) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            current_state: [x, y, theta, vx, vy, omega]
            target_state:  [x_d, y_d, theta_d]
            target_vel:    [vx_d, vy_d, omega_d]
            target_accel:  [ax_d, ay_d, alpha_d]
        """
        # --- 1. Calculate Errors (e, e_dot) ---
        q = current_state[:3]
        q_dot = current_state[3:]
        
        e = target_state - q
        # Unwrap angle error
        e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
        
        e_dot = target_vel - q_dot
        
        # u = M_hat * (q_ddot_d + Kd*e_dot + Kp*e)
        # This is the "Certainty Equivalence" controller
        
        # The term in the brackets is the "Reference Acceleration"
        ref_accel = target_accel + (self.Kd * e_dot) + (self.Kp * e)
        
        # Build Estimated Mass Matrix
        M_hat = np.diag([self.theta_hat[0], self.theta_hat[0], self.theta_hat[1]])
        
        # Compute commanded Force/Torque
        F_cmd = M_hat @ ref_accel
        
        # We need Y such that: Y * theta = M * ref_accel
        # theta = [m, I]
        # Y must be 3x2 matrix
        Y = np.zeros((3, 2))
        Y[0, 0] = ref_accel[0] # x-accel scales with Mass
        Y[1, 0] = ref_accel[1] # y-accel scales with Mass
        Y[2, 1] = ref_accel[2] # angular-accel scales with Inertia
        
        # dot_theta = Gamma * (Weighted_Error) * Y
        # Weighted_Error comes from x^T P b
        
        update_signal = np.zeros(3)
        B = np.array([0, 1])
        
        for i in range(3):
            # Error state x = [e, e_dot]
            x_error = np.array([e[i], e_dot[i]])
            
            # The "mixing" term from Lyapunov analysis: x^T * P * B
            # This replaces the sliding surface concept with the rigorous Lyapunov result
            weighted_err = x_error.T @ self.P_matrices[i] @ B
            update_signal[i] = weighted_err
            
        # Compute adaptation rate
        # dot_theta = Gamma * Y.T * update_signal
        # We typically add a regressor filtering or normalization in practice, 
        # but sticking to the pure lecture form:
        theta_dot = self.Gamma * (Y.T @ update_signal)
        
        # Apply Update
        self.theta_hat += theta_dot * self.dt
        
        # Safety Clamping
        self.theta_hat = np.clip(self.theta_hat, self.theta_min, self.theta_max)
        
        # Return F_cmd (Force_x, Force_y, Torque_z) to be used by the 
        # get_accels_in_world or similar inverse dynamics
        return F_cmd, self.theta_hat

    def get_accels_for_robot(self, F_cmd):
        # Convert the Calculated Forces back to Accels using the ESTIMATED parameters
        # This is what you pass to the robot's 'get_angles_and_torques'
        # if it expects acceleration inputs.
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        ax = F_cmd[0] / self.theta_hat[0]
        ay = F_cmd[1] / self.theta_hat[0]
        alpha = F_cmd[2] / self.theta_hat[1]
        return np.array([ax, ay, alpha])
    
    
    
if __name__ == "__main__":
    import json
    import os
    
    # Load robot parameters
    config_path = os.path.join(os.path.dirname(__file__), '../../Scene/Configuration.json')
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        robot_cfg = config["Robot"]
    else:
        # Fallback if config file isn't found in this specific path structure
        class MockRobot:
            """TODO: Class docstring.

            Attributes:
                TODO: describe attributes
            """
            mass = 10.0
            inertia = 1.0
            wheel_radius = 0.1
            length = 0.5
            width = 0.5
            max_wheel_rotation_speed = 10.0
            max_wheel_torque = 10.0
            max_steering_speed = 10.0
            max_steering_acceleration = 10.0
        robot = MockRobot()
        robot_cfg = {} # Not used if mock is used
        
    from Scene.Robot import Robot
    real_robot = Robot(robot_cfg)
    actuator = ActuatorController(real_robot)
    reference_generator: SimpleReferenceGenerator = SimpleReferenceGenerator()

    # Tuning
    lambda_gains = np.array([2.0, 2.0, 2.0]) # Reduced slightly for stability
    k_gains = np.array([5.0, 5.0, 5.0])      # Reduced K to minimize chattering
    controller = AdaptiveController(actuator, reference_generator=reference_generator, lambda_gains=lambda_gains, k_gains=k_gains)

    # Initial State
    p0 = np.zeros(10)
    p0[:6] = [0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0] # Start at origin, facing 90 deg (North)

    # Simulation
    T = 10.0
    dt = 0.05
    steps = int(T / dt)
    states = np.zeros((steps + 1, 10))
    ref_states = np.zeros((steps + 1, 6))
    times = np.linspace(0, T, steps + 1)
    states[0] = p0.copy()
    ref_states[0, :3] = reference_generator.get_reference_state(0)

    sim = Robot_Sim(actuator, real_robot, dt=dt)
    sim.set_state(p0)

    print("Starting Simulation...")
    for i in range(steps):
        t = times[i]
        current_theta = states[i][2]
        
        # 1. Get GLOBAL accelerations from SMC
        state6d = states[i][:6]
        control_vec = controller.get_command(state6d, t)

        
        # 5. Propagate
        next_state = sim.propagate(states[i], control_vec)
        states[i + 1] = next_state
        ref_states[i + 1, :3] = reference_generator.get_reference_state(t + dt)

    # Dynamic visualization over time
    import matplotlib.pyplot as plt
    import time

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('SMC Path Following (Interactive)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')

    # Precompute axis limits from data
    min_x = min(np.min(states[:, 0]), np.min(ref_states[:, 0])) - 5
    max_x = max(np.max(states[:, 0]), np.max(ref_states[:, 0])) + 5
    min_y = min(np.min(states[:, 1]), np.min(ref_states[:, 1])) - 5
    max_y = max(np.max(states[:, 1]), np.max(ref_states[:, 1])) + 5
    ax.set_xlim(min_x, max_x) # type: ignore
    ax.set_ylim(min_y, max_y) # type: ignore

    robot_line, = ax.plot([], [], 'b-', label='Robot Trajectory')
    ref_line, = ax.plot([], [], 'r--', label='Target')
    robot_point, = ax.plot([], [], 'bo', label='Robot')
    ref_point, = ax.plot([], [], 'ro', label='Target')
    # Draw start heading arrow once
    start_arrow = ax.arrow(states[0, 0], states[0, 1], math.cos(states[0, 2]), math.sin(states[0, 2]), head_width=0.5, color='g')
    ax.legend()

    # Animate by updating line data
    for i in range(len(times)):
        # update trajectory lines
        robot_line.set_data(states[:i+1, 0], states[:i+1, 1])
        ref_line.set_data(ref_states[:i+1, 0], ref_states[:i+1, 1])
        # update current points (wrap in lists)
        robot_point.set_data([states[i, 0]], [states[i, 1]])
        ref_point.set_data([ref_states[i, 0]], [ref_states[i, 1]])

        fig.canvas.draw()
        fig.canvas.flush_events()

        # pace to real time using simulation dt (if available)
        try:
            time.sleep(dt)
        except NameError:
            # fallback: small pause
            time.sleep(0.02)

    plt.ioff()
    # show final static figure (blocking)
    plt.show()

    # --- Post-simulation error analysis and plots ---
    # Compute 6D errors: [ex, ey, eth, evx, evy, evth] as reference - robot
    N = states.shape[0]
    errors = np.zeros((N, 6))
    for i in range(N):
        ref = ref_states[i]
        rob_pos = states[i, :3]
        rob_vel = states[i, 3:6]
        # reference vector: [x,y,theta,vx,vy,vth]
        ref_full = np.zeros(6)
        ref_full[:3] = ref[:3]
        ref_full[3:] = ref[3:]
        err = ref_full - np.concatenate([rob_pos, rob_vel])
        # angle wrap for theta error
        err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
        errors[i] = err

    # Phase portraits (e vs e_dot) for x, y, theta
    import matplotlib.pyplot as plt
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['x error phase', 'y error phase', 'theta error phase']
    labels = ['error', 'error rate']
    comps = [(0, 3), (1, 4), (2, 5)]
    for ax, (p_idx, v_idx), title in zip(axes, comps, titles):
        ax.plot(errors[:, p_idx], errors[:, v_idx], '-', linewidth=1)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Errors over time (all 6)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    names = ['ex', 'ey', 'eth', 'evx', 'evy', 'evth']
    for i in range(6):
        ax3.plot(times, errors[:, i], label=names[i])
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error')
    ax3.set_title('Tracking Errors over Time')
    ax3.legend()
    ax3.grid(True)
    plt.show()