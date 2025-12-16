import math
import sys
from pathlib import Path

# Ensure project root (SwerveVehicle) is on sys.path so top-level packages import correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # PointFollowing->PathController->SwerveVehicle

from Uncertainties.uncertainty import add_force_uncertainty

from PathController.LocalPlanners.PathReference import ReferenceGenerator, SimpleReferenceGenerator
# Add project root to sys.path so imports work from anywhere

from typing import TypeAlias
from ActuatorController.ActuatorController import ActuatorController
from PathController.Robot_Sim import Robot_Sim

import numpy as np
from Types import State6D
import json
import os
from Scene.Robot import Robot

    

State: TypeAlias = np.ndarray #(3, )
'''np (3,) x, y, theta'''

def get_state_and_vel_error(current_state: State6D, reference_state: State6D) -> State6D:
    return reference_state - current_state

class SMCController:
    def __init__(self, robot_controller: ActuatorController, reference_generator: SimpleReferenceGenerator, lambda_gains: np.ndarray, k_gains: np.ndarray, boundary_layer: float = 0.1):
        self.actuator: ActuatorController = robot_controller
        self.reference_generator: SimpleReferenceGenerator = reference_generator


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

    def get_command(self, state: State6D, t: float) -> np.ndarray:
        # --- 1. State Conversion (Robot -> Global) ---
        x, y, theta = state[0], state[1], state[2]
        vx_R, vy_R, v_theta = state[3], state[4], state[5]
        
        c, s_ang = np.cos(theta), np.sin(theta)
        vx_G: float = c * vx_R - s_ang * vy_R
        vy_G: float = s_ang * vx_R + c * vy_R
        
        current_global_pos = np.array([x, y, theta])
        current_global_vel = np.array([vx_G, vy_G, v_theta])

        # --- 2. Get Reference (The "Carrot") ---
        # ref_state: [x_r, y_r, th_r, vx_r, vy_r, vth_r]
        
        ref_pos = self.reference_generator.get_reference_state(t)
        ref_vel = self.reference_generator.get_reference_velocity(t)
        
        # For this implementation, we assume constant speed segments (ref_accel = 0)
        # If your path generator supported curvature acceleration, you would add it here.
        ref_accel: State = self.reference_generator.get_reference_acceleration(t)

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
        control_vec = np.zeros(8)
        control_vec[0:4] = torques
        control_vec[4:8] = deltas
        
        return control_vec
    
if __name__ == "__main__":
    # Load robot parameters
    config_path = os.path.join(os.path.dirname(__file__), '../../Scene/Configuration.json')
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        robot_cfg = config["Robot"]
    else:
        # Fallback if config file isn't found in this specific path structure
        # (Mocking values for standalone test)
        class MockRobot:
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
        
    real_robot = Robot(robot_cfg)
    actuator = ActuatorController(real_robot)
    reference_generator: SimpleReferenceGenerator = SimpleReferenceGenerator()

    # Tuning
    lambda_gains = np.array([2.0, 2.0, 2.0]) # Reduced slightly for stability
    k_gains = np.array([5.0, 5.0, 5.0])      # Reduced K to minimize chattering
    controller = SMCController(actuator, reference_generator=reference_generator, lambda_gains=lambda_gains, k_gains=k_gains)

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
        noise = add_force_uncertainty(5.0, 5.0, 2)  # Example disturbance noise

        
        # 5. Propagate
        next_state = sim.propagate(states[i], control_vec, noise=noise)
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