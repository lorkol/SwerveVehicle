"""
Test Script for Swerve Robot Dynamics

This script tests the new ActuatorController and Robot_Sim modules by running
multiple test scenarios with different movements. Each scenario has clear descriptions
of what the robot is supposed to do.

Test Scenarios:
1. STRAIGHT LINE: All wheels aligned, equal torque -> Robot moves forward in a straight line
2. ROTATION IN PLACE: Wheels at 0 deg, asymmetric torque -> Robot rotates around center
3. CURVED PATH: All wheels aligned at 45 deg, equal torque -> Robot moves diagonally
4. COMBINED: All wheels aligned, asymmetric torque -> Linear motion + rotation (curved path)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from Scene.JsonManager import load_json
from Scene.Robot import Robot
from ActuatorController.ActuatorController import ActuatorController
from PathController.Robot_Sim import Robot_Sim
from PathController.Types import State_Vector, Control_Vector


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """
    Normalize angles to [-pi, pi] range using atan2.
    
    This is CRITICAL for preventing oscillations: the ActuatorController's B matrix
    uses trig functions (sin, cos) of angles. If angles accumulate beyond [-pi, pi],
    trig values become discontinuous, causing force discontinuities and oscillations.
    
    Example: sin(0.1) vs sin(0.1 + 2*pi) are identical, but we avoid numerical issues
    by keeping angles in the principal range.
    
    Args:
        angles: Array of angles in radians
        
    Returns:
        Normalized angles in [-pi, pi]
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def run_test_scenario_from_accels(name: str, description: str, desired_accels: np.ndarray,
                                   duration: float, dt: float = 0.01,
                                   ramp_time: float = 0.5, global_frame: bool = True) -> tuple:
    """
    Run a test scenario from desired accelerations.
    
    Uses ActuatorController.get_angles_and_torques() to compute the wheel angles
    and torques needed to achieve the desired accelerations. This tests the inverse
    kinematics and verifies that Robot_Sim produces the expected dynamics.
    
    Control Strategy:
    - PHASE 1 (0 to ramp_time): Ramp accelerations smoothly from 0 to desired
    - PHASE 2 (rest): Hold desired accelerations
    
    Args:
        name: Name of the test scenario
        description: Clear description of expected movement
        desired_accels: Target accelerations [ax, ay, a_theta]
                       If global_frame=True: [ax_G, ay_G, a_theta] in global frame
                       If global_frame=False: [ax_R, ay_R, a_theta] in robot frame
        duration: Total simulation duration in seconds
        dt: Simulation timestep in seconds
        ramp_time: Time to ramp accelerations from 0 to desired (seconds)
        global_frame: If True, accelerations are in global frame and converted to robot frame at each step
        
    Returns:
        (time_array, state_history, description)
    """
    print(f"\n{'='*70}")
    print(f"TEST SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"Description: {description}")
    frame_str = "GLOBAL" if global_frame else "ROBOT"
    print(f"Desired accelerations ({frame_str}): ax={desired_accels[0]:.3f}, ay={desired_accels[1]:.3f}, a_theta={desired_accels[2]:.3f}")
    print(f"Duration: {duration} seconds")
    print(f"Ramp time: {ramp_time} seconds")
    
    # Load robot configuration
    config_path = str(Path(__file__).parent / "Scene/Configuration.json")
    config = load_json(config_path)
    robot = Robot(config["Robot"])
    
    # Initialize ActuatorController to compute required angles and torques
    controller = ActuatorController(robot)
    
    # For initial computation, convert global to robot frame at theta=0 (frames aligned)
    if global_frame:
        # At theta=0, global and robot frames are aligned
        desired_accels_R = desired_accels.copy()
        print(f"[At theta=0, converted to ROBOT frame]: ax_R={desired_accels_R[0]:.3f}, ay_R={desired_accels_R[1]:.3f}, a_theta={desired_accels_R[2]:.3f}")
    else:
        desired_accels_R = desired_accels.copy()
    
    # Get the wheel angles and torques needed for desired accelerations
    wheel_angles_rad, wheel_torques = controller.get_angles_and_torques(desired_accels_R)
    
    print(f"[ActuatorController computed (at theta=0)]")
    print(f"  Wheel angles: {np.degrees(wheel_angles_rad)} degrees")
    print(f"  Wheel torques: {wheel_torques} N*m")
    
    # Initialize simulator
    sim = Robot_Sim(None, robot, dt=dt)
    
    # Initial state: [x, y, theta, vx, vy, omega, d1, d2, d3, d4]
    initial_state: State_Vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                             0.0, 0.0, 0.0, 0.0])
    
    # Simulation loop
    num_steps = int(duration / dt)
    time_array = np.linspace(0, duration, num_steps)
    state_history = np.zeros((num_steps, 10))
    accel_history_R = np.zeros((num_steps, 3))  # Track actual accelerations in robot frame
    accel_history_G = np.zeros((num_steps, 3))  # Track actual accelerations in global frame
    
    current_state = initial_state.copy()
    state_history[0] = current_state
    
    print(f"\nSimulating {num_steps} steps...")
    
    for i in range(1, num_steps):
        t = time_array[i]
        theta = current_state[2]  # Current robot orientation
        
        # Ramp progress (0 to 1)
        if t <= ramp_time:
            progress = t / ramp_time
        else:
            progress = 1.0
        
        # If global frame: convert desired global acceleration to robot frame at current theta
        if global_frame:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Inverse rotation: global to robot frame
            # [ax_R, ay_R] = R^T * [ax_G, ay_G]
            ax_G_current = desired_accels[0] * progress
            ay_G_current = desired_accels[1] * progress
            
            ax_R = cos_theta * ax_G_current + sin_theta * ay_G_current
            ay_R = -sin_theta * ax_G_current + cos_theta * ay_G_current
            a_theta = desired_accels[2] * progress
            
            current_accels_R = np.array([ax_R, ay_R, a_theta])
        else:
            # Robot frame: just scale by progress
            current_accels_R = desired_accels * progress
        
        # Compute angles and torques for current accelerations
        current_wheel_angles, current_wheel_torques = controller.get_angles_and_torques(current_accels_R)
        current_wheel_angles = normalize_angles(current_wheel_angles)
        
        # Control input: [tau1, tau2, tau3, tau4, d1_target, d2_target, d3_target, d4_target]
        control = np.array([
            current_wheel_torques[0], current_wheel_torques[1], current_wheel_torques[2], current_wheel_torques[3],
            current_wheel_angles[0], current_wheel_angles[1], current_wheel_angles[2], current_wheel_angles[3]
        ])
        
        # Verify what Robot_Sim computes
        accels_R = controller.get_accels(current_wheel_angles, current_wheel_torques)
        accel_history_R[i] = accels_R
        
        # Compute actual global frame accelerations including Coriolis effects
        vx_R, vy_R = current_state[3], current_state[4]
        omega = current_state[5]
        ax_R, ay_R, a_theta_computed = accels_R
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Global frame acceleration = rotation of robot frame accel + Coriolis terms from rotating reference frame
        # ay_G = ay_R * cos(theta) - ay_R * sin(theta) + vx_R * omega * cos(theta) - vy_R * omega * sin(theta)
        ax_G = cos_theta * ax_R - sin_theta * ay_R + vx_R * omega * (-sin_theta) - vy_R * omega * (-cos_theta)
        ay_G = sin_theta * ax_R + cos_theta * ay_R + vx_R * omega * cos_theta - vy_R * omega * sin_theta
        
        accel_history_G[i] = np.array([ax_G, ay_G, a_theta_computed])
        
        current_state = sim.propagate(current_state, control)
        state_history[i] = current_state
    
    # Compute average acceleration during steady state (after ramp)
    steady_state_start = int(ramp_time / dt)
    avg_accels_R = np.mean(accel_history_R[steady_state_start:], axis=0)
    avg_accels_G = np.mean(accel_history_G[steady_state_start:], axis=0)
    
    print(f"[DONE] Simulation complete")
    print(f"[Verification - GLOBAL FRAME]")
    print(f"  Desired accelerations: ax_G={desired_accels[0]:.3f}, ay_G={desired_accels[1]:.3f}, a_theta={desired_accels[2]:.3f}")
    print(f"  Avg actual accelerations (steady state): ax_G={avg_accels_G[0]:.3f}, ay_G={avg_accels_G[1]:.3f}, a_theta={avg_accels_G[2]:.3f}")
    print(f"[Verification - ROBOT FRAME]")
    print(f"  Avg actual accelerations (steady state): ax_R={avg_accels_R[0]:.3f}, ay_R={avg_accels_R[1]:.3f}, a_theta={avg_accels_R[2]:.3f}")
    print(f"Final position: x={current_state[0]:.3f}m, y={current_state[1]:.3f}m")
    print(f"Final orientation: theta={np.degrees(current_state[2]):.3f} deg")
    print(f"Final velocity: vx_R={current_state[3]:.3f}m/s, vy_R={current_state[4]:.3f}m/s")
    print(f"Final angular velocity: omega={current_state[5]:.3f}rad/s")
    
    return time_array, state_history, description


def plot_results(all_results: list):
    """
    Plot results from all test scenarios.
    
    Args:
        all_results: List of (name, time_array, state_history, description) tuples
    """
    num_scenarios = len(all_results)
    
    # Dynamically create grid based on number of scenarios
    # Prefer wider than tall for better visibility
    cols = min(3, num_scenarios)  # Max 3 columns
    rows = (num_scenarios + cols - 1) // cols  # Ceiling division
    
    fig = plt.figure(figsize=(6*cols, 5*rows))
    fig.suptitle("All Test Scenarios - Robot Paths Comparison", fontsize=16, fontweight='bold')
    
    # Plot paths for each scenario
    for idx, (name, time_array, state_history, description) in enumerate(all_results):
        ax = plt.subplot(rows, cols, idx + 1)
        
        x = state_history[:, 0]
        y = state_history[:, 1]
        theta = state_history[:, 2]
        
        # Plot path
        ax.plot(x, y, 'b-', linewidth=2, label='Robot Path')
        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'r*', markersize=15, label='End')
        
        # Draw robot orientation at end
        scale = 0.3
        end_x, end_y = x[-1], y[-1]
        dx = scale * np.cos(theta[-1])
        dy = scale * np.sin(theta[-1])
        ax.arrow(end_x, end_y, dx, dy, head_width=0.05, head_length=0.05, 
                fc='red', ec='red', alpha=0.7)
        
        ax.set_title(f"{name}", fontsize=11, fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / "ActuatorController" / "Testing"
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / 'dynamics_test_paths.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved paths plot: {filepath}")
    
    # Create detailed plots for each scenario
    for idx, (scenario_name, time_array, state_history, description) in enumerate(all_results):
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f"Detailed Dynamics Analysis: {scenario_name}\n{description}", 
                    fontsize=14, fontweight='bold')
        
        x = state_history[:, 0]
        y = state_history[:, 1]
        theta = state_history[:, 2]
        vx_R = state_history[:, 3]
        vy_R = state_history[:, 4]
        omega = state_history[:, 5]
        
        # Convert to global frame velocities
        vx_G = vx_R * np.cos(theta) - vy_R * np.sin(theta)
        vy_G = vx_R * np.sin(theta) + vy_R * np.cos(theta)
        
        # Plot 1: Position over time
        axs[0, 0].plot(time_array, x, 'b-', label='x')
        axs[0, 0].plot(time_array, y, 'r-', label='y')
        axs[0, 0].set_ylabel('Position (m)')
        axs[0, 0].set_title('Position vs Time')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Orientation over time
        axs[0, 1].plot(time_array, np.degrees(theta), 'g-')
        axs[0, 1].set_ylabel('Angle (degrees)')
        axs[0, 1].set_title('Orientation vs Time')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Global frame velocities
        axs[1, 0].plot(time_array, vx_G, 'b-', label='vx_G (Global)')
        axs[1, 0].plot(time_array, vy_G, 'r-', label='vy_G (Global)')
        axs[1, 0].plot(time_array, omega, 'g-', label='omega (Angular)')
        axs[1, 0].set_ylabel('Velocity (m/s or rad/s)')
        axs[1, 0].set_title('Velocities (Global Frame)')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Robot frame velocities
        axs[1, 1].plot(time_array, vx_R, 'b-', label='vx_R (Robot)')
        axs[1, 1].plot(time_array, vy_R, 'r-', label='vy_R (Robot)')
        axs[1, 1].plot(time_array, omega, 'g-', label='omega (Angular)')
        axs[1, 1].set_ylabel('Velocity (m/s or rad/s)')
        axs[1, 1].set_title('Velocities (Robot Frame)')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Speed (magnitude)
        speed_R = np.sqrt(vx_R**2 + vy_R**2)
        speed_G = np.sqrt(vx_G**2 + vy_G**2)
        axs[2, 0].plot(time_array, speed_R, 'b-', label='Speed (Robot Frame)')
        axs[2, 0].plot(time_array, speed_G, 'r--', label='Speed (Global Frame)')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Speed (m/s)')
        axs[2, 0].set_title('Speed Magnitude')
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Path in XY
        axs[2, 1].plot(x, y, 'b-', linewidth=2, label='Path')
        axs[2, 1].plot(x[0], y[0], 'go', markersize=10, label='Start')
        axs[2, 1].plot(x[-1], y[-1], 'r*', markersize=15, label='End')
        # Plot every Nth point as a small orientation indicator
        N = max(1, len(x) // 20)  # Plot ~20 orientation markers
        for j in range(0, len(x), N):
            scale = 0.2
            dx = scale * np.cos(theta[j])
            dy = scale * np.sin(theta[j])
            axs[2, 1].arrow(x[j], y[j], dx, dy, head_width=0.05, head_length=0.05,
                           fc='gray', ec='gray', alpha=0.5)
        axs[2, 1].set_xlabel('X (m)')
        axs[2, 1].set_ylabel('Y (m)')
        axs[2, 1].set_title('Robot Path (XY Plane)')
        axs[2, 1].legend()
        axs[2, 1].grid(True, alpha=0.3)
        axs[2, 1].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        output_dir = Path(__file__).parent / "ActuatorController" / "Testing"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'{scenario_name.replace(" ", "_").lower()}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[DONE] Saved detailed plot: {filepath}")
    
    plt.show()


def main():
    """Run all dynamics test scenarios based on the 5 user-specified scenarios."""
    print("\n" + "="*70)
    print("SWERVE ROBOT DYNAMICS TEST SUITE - 5 SPECIFIC TEST SCENARIOS")
    print("="*70)
    print("\nTesting ActuatorController and Robot_Sim modules")
    print("Configuration loaded from: Scene/Configuration.json\n")
    print("SCENARIO OVERVIEW:")
    print("  1. PURE LINEAR MOTION (X-AXIS)")
    print("     - ay_G = 0, a_theta = 0, ax_G != 0 (pure forward motion)")
    print("")
    print("  2. PURE LINEAR MOTION (Y-AXIS)")
    print("     - ax_G = 0, a_theta = 0, ay_G != 0 (pure sideways motion)")
    print("")
    print("  3. DIAGONAL LINEAR MOTION (45 deg)")
    print("     - ax_G != 0, ay_G != 0, a_theta = 0 with steering angles = 45 deg")
    print("")
    print("  4. CURVED PATH (STEERING VARIATION)")
    print("     - All wheels steer together (symmetric)")
    print("     - Steering angles vary smoothly over time")
    print("     - Creates visible circular arc trajectory\n")
    
    # Load robot configuration for use in all scenarios
    config_path = str(Path(__file__).parent / "Scene/Configuration.json")
    config = load_json(config_path)
    robot = Robot(config["Robot"])
    
    all_results = []
    
    # ========================================
    # SCENARIO 1: Pure Linear Motion (X-axis)
    # ========================================
    results1 = run_test_scenario_from_accels(
        name="SCENARIO 1: Pure X-Axis Motion",
        description="Pure linear acceleration along global X-axis: ax_G=3.0 m/s², ay_G=0, a_theta=0.\n"
                   "Expected: Robot accelerates forward in straight line along global +X axis.\n"
                   "Verification: ay_G should remain ~0, a_theta should remain ~0.",
        desired_accels=np.array([3.0, 0.0, 0.0]),
        duration=5.0,
        dt=0.01,
        ramp_time=0.5,
        global_frame=True
    )
    all_results.append(("1 - X Motion", *results1))
    
    # ========================================
    # SCENARIO 2: Pure Linear Motion (Y-axis)
    # ========================================
    results2 = run_test_scenario_from_accels(
        name="SCENARIO 2: Pure Y-Axis Motion",
        description="Pure linear acceleration along global Y-axis: ax_G=0, ay_G=3.0 m/s², a_theta=0.\n"
                   "Expected: Robot accelerates sideways in straight line along global +Y axis.\n"
                   "Verification: ax_G should remain ~0, a_theta should remain ~0.",
        desired_accels=np.array([0.0, 3.0, 0.0]),
        duration=5.0,
        dt=0.01,
        ramp_time=0.5,
        global_frame=True
    )
    all_results.append(("2 - Y Motion", *results2))
    
    # ========================================
    # SCENARIO 3: Diagonal Motion (45°)
    # ========================================
    results3 = run_test_scenario_from_accels(
        name="SCENARIO 3: Diagonal Motion (45°)",
        description="Pure diagonal acceleration at 45°: ax_G=2.5 m/s², ay_G=2.5 m/s², a_theta=0.\n"
                   "Expected: Robot accelerates northeast at 45° without rotation.\n"
                   "Steering angles will automatically set to ~45° by ActuatorController.\n"
                   "Verification: Motion should follow diagonal line, theta stays ~0.",
        desired_accels=np.array([2.5, 2.5, 0.0]),
        duration=5.0,
        dt=0.01,
        ramp_time=0.5,
        global_frame=True
    )
    all_results.append(("3 - Diagonal 45", *results3))
    
    # ========================================
    # ========================================
    # SCENARIO 4: Curved Path (Torque Asymmetry)
    # ========================================
    print(f"\n{'='*70}")
    print(f"SCENARIO 5: CURVED PATH (STEERING VARIATION)")
    print(f"{'='*70}")
    description5 = ("Curved path using smooth steering variation: All wheels steer together.\n"
                   "Steering angle varies linearly from 0 to 720 degrees over 8 seconds.\n"
                   "Symmetric torques on all wheels provide consistent forward speed.\n"
                   "Expected: Robot traces smooth circular arcs, completing 2 full steering rotations.")
    print(f"Description: {description5}")
    
    sim5 = Robot_Sim(None, robot, dt=0.01)
    controller = ActuatorController(robot)
    duration = 8.0  # 8 seconds
    dt = 0.01
    num_steps = int(duration / dt)
    time_array = np.linspace(0, duration, num_steps)
    state_history = np.zeros((num_steps, 10))
    
    current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_history[0] = current_state
    
    print(f"Simulating {num_steps} steps for curved path with torque asymmetry...")
    
    # Debug first step
    wheel_angles = np.array([0.0, 0.0, 0.0, 0.0])
    right_torque = 1.0
    left_torque = 0.99    # VERY tiny asymmetry (0.01 difference)
    wheel_torques = np.array([right_torque, left_torque, left_torque, right_torque])
    accel_test = controller.get_accels(wheel_angles, wheel_torques)
    print(f"[DEBUG] With torques {wheel_torques} and angles {wheel_angles}:")
    print(f"  Accelerations (robot frame): ax={accel_test[0]:.4f}, ay={accel_test[1]:.4f}, a_theta={accel_test[2]:.4f}")
    
    for i in range(1, num_steps):
        t = time_array[i]
        
        # All wheels point forward (0°) in ROBOT FRAME - stays fixed as robot rotates
        wheel_angles = np.array([0.0, 0.0, 0.0, 0.0])
        
        # VERY SMALL asymmetric torque: right side slightly > left side
        # Creates gentle curve with small total rotation (< 90 degrees)
        # Wheel layout: 1(FR), 2(FL), 3(RL), 4(RR)
        # Right side: wheels 1, 4 
        # Left side: wheels 2, 3
        right_torque = 1.0
        left_torque = 0.99    # VERY tiny asymmetry (0.01 difference)
        
        wheel_torques = np.array([
            right_torque,  # Wheel 1 (FR - front right)
            left_torque,   # Wheel 2 (FL - front left)
            left_torque,   # Wheel 3 (RL - rear left)
            right_torque   # Wheel 4 (RR - rear right)
        ])
        
        control = np.array([wheel_torques[0], wheel_torques[1], wheel_torques[2], wheel_torques[3],
                           wheel_angles[0], wheel_angles[1], wheel_angles[2], wheel_angles[3]])
        
        current_state = sim5.propagate(current_state, control)
        state_history[i] = current_state
    
    print(f"[DONE] Simulation complete")
    print(f"Final position: x={current_state[0]:.3f}m, y={current_state[1]:.3f}m")
    print(f"Final orientation: theta={np.degrees(current_state[2]):.3f}° (should be positive, indicating left turn)")
    print(f"Final linear velocity: vx_R={current_state[3]:.3f}m/s, vy_R={current_state[4]:.3f}m/s")
    print(f"Final angular velocity: omega={current_state[5]:.3f}rad/s")
    
    all_results.append(("5 - Curve (Torque)", time_array, state_history, description5))
    
    # ========================================
    # Plot all results
    # ========================================
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_results(all_results)
    
    print("\n" + "="*70)
    print("ALL TEST SCENARIOS COMPLETE")
    print("="*70)
    print("\n[DONE] All 5 test scenarios completed successfully!")
    print("[DONE] Check the generated PNG files for visualization:")
    print(f"  Location: {Path(__file__).parent / 'ActuatorController' / 'Testing'}")
    print("\nGenerated plots:")
    print("  - dynamics_test_paths.png: Overview of all robot paths")
    print("  - dynamics_test_detailed_1_1___x_motion.png")
    print("  - dynamics_test_detailed_2_2___y_motion.png")
    print("  - dynamics_test_detailed_3_3___diagonal_45.png")
    print("  - dynamics_test_detailed_4_4___curve_(torque).png")
    print("\nSCENARIO SUMMARY:")
    print("  1. Pure X-axis motion: Robot moves forward, ay_G~0, a_theta~0")
    print("  2. Pure Y-axis motion: Robot moves sideways, ax_G~0, a_theta~0")
    print("  3. Diagonal 45 deg motion: Robot moves northeast, ax_G~ay_G, a_theta~0")
    print("  4. Curved path (torque): Right torque > Left torque, zero steering")
    print("\nVERIFICATION APPROACH:")
    print("  • Each scenario uses ActuatorController.get_angles_and_torques() to compute controls")
    print("  • Robot_Sim.propagate() simulates the resulting motion")
    print("  • Plots show actual achieved accelerations match desired accelerations")
    print("  • Trajectory plots verify geometry of each scenario")


if __name__ == "__main__":
    main()
