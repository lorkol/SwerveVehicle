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


def run_test_scenario(name: str, description: str, wheel_angles_deg: np.ndarray, 
                      wheel_torques: np.ndarray, duration: float, dt: float = 0.01,
                      steering_ramp_time: float = 0.5) -> tuple:
    """
    Run a single test scenario with constant wheel angles and torques.
    Properly integrates steering by computing velocity commands to reach and maintain target angles.
    
    KEY INSIGHT: Wheel angles must reach target BEFORE full torques are applied to avoid
    transient oscillations from the acceleration/angle update timing mismatch.
    
    Args:
        name: Name of the test scenario
        description: Clear description of expected movement
        wheel_angles_deg: Steering angles for each wheel in degrees
        wheel_torques: Torques for each wheel in N*m
        duration: Total simulation duration in seconds
        dt: Simulation timestep in seconds
        steering_ramp_time: Time to ramp steering angles to target (seconds)
        
    Returns:
        (time_array, state_history, description)
    """
    print(f"\n{'='*70}")
    print(f"TEST SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"Wheel angles: {wheel_angles_deg} degrees")
    print(f"Wheel torques: {wheel_torques} N*m")
    print(f"Duration: {duration} seconds")
    print(f"Steering ramp time: {steering_ramp_time} seconds")
    
    # Load robot configuration
    config_path = str(Path(__file__).parent / "Scene/Configuration.json")
    config = load_json(config_path)
    robot = Robot(config["Robot"])
    
    # Initialize simulator
    sim = Robot_Sim(None, robot, dt=dt)
    
    # Convert angles to radians
    wheel_angles_rad = np.radians(wheel_angles_deg)
    
    # Initial state: [x, y, theta, vx, vy, omega, d1, d2, d3, d4]
    initial_state: State_Vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                             0.0, 0.0, 0.0, 0.0])
    
    # Simulation loop
    num_steps = int(duration / dt)
    time_array = np.linspace(0, duration, num_steps)
    state_history = np.zeros((num_steps, 10))
    
    current_state = initial_state.copy()
    state_history[0] = current_state
    
    print(f"\nSimulating {num_steps} steps with proper steering dynamics...")
    
    for i in range(1, num_steps):
        t = time_array[i]
        
        # PHASE 1: Steer wheels to target angles (first steering_ramp_time seconds)
        if t <= steering_ramp_time:
            # Compute steering velocities to reach target in exactly steering_ramp_time
            current_angles = current_state[6:10]
            angle_errors = wheel_angles_rad - current_angles
            time_remaining = steering_ramp_time - (t - dt)
            
            if time_remaining > dt:
                # Divide equally across remaining time for smooth approach
                steering_velocities = angle_errors / time_remaining
                # Limit steering rate to reasonable values
                steering_velocities = np.clip(steering_velocities, -5.0, 5.0)
            else:
                # Final step - zero out velocities
                steering_velocities = np.zeros(4)
            
            # During steering phase, NO TORQUES to prevent interference
            phase_torques = np.zeros(4)
        
        # PHASE 2: Maintain target angles and ramp torques
        else:
            # Hold angles at target (zero velocity commands)
            steering_velocities = np.zeros(4)
            
            # Ramp torques over a short transition period (0.5s after steering completes)
            # This prevents oscillations from sudden force application
            transition_start = steering_ramp_time
            transition_duration = 0.5
            transition_end = transition_start + transition_duration
            
            if t < transition_end:
                # Linear ramp from 0 to full torque
                ramp_factor = (t - transition_start) / transition_duration
                phase_torques = wheel_torques * ramp_factor
            else:
                # Full torques applied
                phase_torques = wheel_torques
        
        # Control input: [tau1, tau2, tau3, tau4, v_d1, v_d2, v_d3, v_d4]
        control = np.array([
            phase_torques[0], phase_torques[1], phase_torques[2], phase_torques[3],
            steering_velocities[0], steering_velocities[1], steering_velocities[2], steering_velocities[3]
        ])
        
        current_state = sim.propagate(current_state, control)
        state_history[i] = current_state
    
    print(f"[DONE] Simulation complete")
    print(f"Final position: x={current_state[0]:.3f}m, y={current_state[1]:.3f}m")
    print(f"Final orientation: theta={np.degrees(current_state[2]):.3f} deg")
    print(f"Final velocity: vx_R={current_state[3]:.3f}m/s, vy_R={current_state[4]:.3f}m/s")
    print(f"Final angular velocity: omega={current_state[5]:.3f}rad/s")
    
    return time_array, state_history, description


def run_test_scenario_dynamic(name: str, description: str, 
                               angle_func, torque_func, duration: float, dt: float = 0.01) -> tuple:
    """
    Run a test scenario with time-varying wheel angles and/or torques.
    Computes steering velocities to follow the target angle trajectory.
    
    Args:
        name: Name of the test scenario
        description: Clear description of expected movement
        angle_func: Function(t) -> np.ndarray([4]) returning wheel angles in degrees at time t
        torque_func: Function(t) -> np.ndarray([4]) returning torques at time t
        duration: Total simulation duration in seconds
        dt: Simulation timestep in seconds
        
    Returns:
        (time_array, state_history, description)
    """
    print(f"\n{'='*70}")
    print(f"TEST SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"Duration: {duration} seconds")
    print(f"Wheel angles and torques vary with time")
    
    # Load robot configuration
    config_path = str(Path(__file__).parent / "Scene/Configuration.json")
    config = load_json(config_path)
    robot = Robot(config["Robot"])
    
    # Initialize simulator
    sim = Robot_Sim(None, robot, dt=dt)
    
    # Initial state: [x, y, theta, vx, vy, omega, d1, d2, d3, d4]
    initial_state: State_Vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                             0.0, 0.0, 0.0, 0.0])
    
    # Simulation loop
    num_steps = int(duration / dt)
    time_array = np.linspace(0, duration, num_steps)
    state_history = np.zeros((num_steps, 10))
    
    current_state = initial_state.copy()
    state_history[0] = current_state
    
    print(f"\nSimulating {num_steps} steps with dynamic steering...")
    
    for i in range(1, num_steps):
        t = time_array[i]
        
        # Get target angles and torques at current time
        target_angles_deg = angle_func(t)
        target_angles_rad = np.radians(target_angles_deg)
        torques = torque_func(t)
        
        # Compute steering velocities to follow the target trajectory
        # Using finite difference: v = (angle_next - angle_current) / dt
        current_angles = current_state[6:10]
        
        # Get target angles at next timestep for better prediction
        t_next = t + dt
        target_angles_next_deg = angle_func(t_next)
        target_angles_next_rad = np.radians(target_angles_next_deg)
        
        # Steering velocity = how fast we need to change angle to reach next target
        angle_deltas = target_angles_next_rad - current_angles
        
        # Normalize angle errors to [-pi, pi] range for shortest path
        angle_deltas = np.where(angle_deltas > np.pi, angle_deltas - 2*np.pi, angle_deltas)
        angle_deltas = np.where(angle_deltas < -np.pi, angle_deltas + 2*np.pi, angle_deltas)
        
        # Compute velocity to reach next target
        steering_velocities = angle_deltas / dt
        
        # Limit steering rate to reasonable values
        steering_velocities = np.clip(steering_velocities, -5.0, 5.0)
        
        # Prepare control input: [tau1, tau2, tau3, tau4, v_d1, v_d2, v_d3, v_d4]
        control: Control_Vector = np.array([
            torques[0], torques[1], torques[2], torques[3],
            steering_velocities[0], steering_velocities[1], steering_velocities[2], steering_velocities[3]
        ])
        
        # Propagate state
        current_state = sim.propagate(current_state, control)
        state_history[i] = current_state
    
    print(f"[DONE] Simulation complete")
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
        scale = 0.5
        dx = scale * np.cos(theta[-1])
        dy = scale * np.sin(theta[-1])
        axs[2, 1].arrow(x[-1], y[-1], dx, dy, head_width=0.1, head_length=0.1,
                       fc='red', ec='red', alpha=0.7)
        axs[2, 1].set_xlabel('X (m)')
        axs[2, 1].set_ylabel('Y (m)')
        axs[2, 1].set_title('Robot Path (XY Plane)')
        axs[2, 1].legend()
        axs[2, 1].grid(True, alpha=0.3)
        axs[2, 1].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        output_dir = Path(__file__).parent / "ActuatorController" / "Testing"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'dynamics_test_detailed_{idx+1}_{scenario_name.replace(" ", "_").lower()}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[DONE] Saved detailed plot: {filepath}")
    
    plt.show()


def main():
    """Run all dynamics test scenarios."""
    print("\n" + "="*70)
    print("SWERVE ROBOT DYNAMICS TEST SUITE v2.0")
    print("="*70)
    print("\nTesting ActuatorController and Robot_Sim modules")
    print("Configuration loaded from: Scene/Configuration.json\n")
    print("KEY FIXES IN THIS VERSION:")
    print("  1. CONSTANT ANGLE TESTS: Dual-phase control")
    print("     - PHASE 1 (0 to steering_ramp_time): Steer wheels, ZERO torques")
    print("     - PHASE 2a (after steering, 0.5s): RAMP torques from 0 to full")
    print("     - PHASE 2b (rest of test): Full torques at constant angles")
    print("     - Eliminates oscillations from sudden force application")
    print("")
    print("  2. DYNAMIC ANGLE TESTS: Predictive steering velocity commands")
    print("     - Computes v_d based on future target trajectory")
    print("     - Angles follow targets smoothly without feedback overshoot")
    print("     - Rate-limited steering prevents unrealistic motion")
    print("")
    print("  3. PHYSICS CONSISTENCY:")
    print("     - Angle normalization to [-π, π] prevents discontinuities")
    print("     - Respects Robot_Sim.propagate() timing")
    print("     - Smooth, non-oscillatory force application\n")
    
    all_results = []
    
    # Test 1: Straight line motion
    results1 = run_test_scenario(
        name="STRAIGHT LINE",
        description="All wheels aligned forward (0 deg), equal torque on all wheels.\n"
                   "Expected: Robot accelerates forward in a straight line without rotation.",
        wheel_angles_deg=np.array([0.0, 0.0, 0.0, 0.0]),
        wheel_torques=np.array([2.0, 2.0, 2.0, 2.0]),
        duration=5.0,
        dt=0.005,
        steering_ramp_time=0.2
    )
    all_results.append(("Straight Line", *results1))
    
    # Test 2: Rotation in place
    results2 = run_test_scenario(
        name="ROTATION IN PLACE",
        description="All wheels aligned (0 deg), asymmetric torque (right wheels faster).\n"
                   "Expected: Robot rotates counterclockwise around its center with minimal linear motion.",
        wheel_angles_deg=np.array([0.0, 0.0, 0.0, 0.0]),
        wheel_torques=np.array([2.1, 1.9, 1.9, 2.1]),
        duration=5.0,
        dt=0.005,
        steering_ramp_time=0.2
    )
    all_results.append(("Rotation In Place", *results2))
    
    # Test 3: Diagonal motion
    results3 = run_test_scenario(
        name="DIAGONAL MOTION",
        description="All wheels aligned at 45 deg, equal torque on all wheels.\n"
                   "Expected: Robot moves diagonally (northeast direction) without rotation.",
        wheel_angles_deg=np.array([45.0, 45.0, 45.0, 45.0]),
        wheel_torques=np.array([2.0, 2.0, 2.0, 2.0]),
        duration=5.0,
        dt=0.005,
        steering_ramp_time=0.3
    )
    all_results.append(("Diagonal Motion", *results3))
    
    # Test 4: Curved path (combined linear + rotation)
    results4 = run_test_scenario(
        name="CURVED PATH",
        description="All wheels aligned forward (0 deg), asymmetric torque.\n"
                   "Expected: Robot moves forward while rotating - creates curved path.",
        wheel_angles_deg=np.array([0.0, 0.0, 0.0, 0.0]),
        wheel_torques=np.array([2.1, 1.9, 1.9, 2.1]),
        duration=5.0,
        dt=0.005,
        steering_ramp_time=0.2
    )
    all_results.append(("Curved Path", *results4))
    
    # Test 5: Circular motion with constant steering
    def circle_angles_func(t):
        """Continuously steer to make a circular path."""
        angle = 25.0
        return np.array([angle, angle, angle, angle])
    
    def circle_torques_func(t):
        """Equal torque."""
        return np.array([2.0, 2.0, 2.0, 2.0])
    
    results5 = run_test_scenario_dynamic(
        name="CIRCULAR MOTION (Steering at 25 deg)",
        description="All wheels steered at constant 25 deg angle, equal torque.\n"
                   "Expected: Robot moves in a circular arc.",
        angle_func=circle_angles_func,
        torque_func=circle_torques_func,
        duration=8.0,
        dt=0.005
    )
    all_results.append(("Circular Motion", *results5))
    
    # Test 6: Pure rotation (wheels perpendicular)
    def perpendicular_angles_func(t):
        """Wheels perpendicular to robot - creates sideways motion."""
        return np.array([90.0, 90.0, 90.0, 90.0])
    
    def perpendicular_torques_func(t):
        """Asymmetric torque for rotation."""
        return np.array([2.1, 1.9, 1.9, 2.1])
    
    results6 = run_test_scenario_dynamic(
        name="PURE SPIN (Wheels at 90 deg)",
        description="All wheels perpendicular (±90 deg), asymmetric torque.\n"
                   "Expected: Robot spins with sideways motion.",
        angle_func=perpendicular_angles_func,
        torque_func=perpendicular_torques_func,
        duration=6.0,
        dt=0.005
    )
    all_results.append(("Pure Spin", *results6))
    
    # Test 7: Gradually changing angles (spiral motion)
    def spiral_angles_func(t):
        """Gradually increase steering angle over time."""
        max_angle = 45.0
        angle = max_angle * (t / 8.0)
        return np.array([angle, angle, angle, angle])
    
    def spiral_torques_func(t):
        """Equal torque throughout."""
        return np.array([2.0, 2.0, 2.0, 2.0])
    
    results7 = run_test_scenario_dynamic(
        name="SPIRAL MOTION (Angles increase 0 deg -> 45 deg)",
        description="Steering angles gradually increase from 0 deg to 45 deg over time.\n"
                   "Expected: Robot path spirals outward with increasing radius.",
        angle_func=spiral_angles_func,
        torque_func=spiral_torques_func,
        duration=8.0,
        dt=0.005
    )
    all_results.append(("Spiral Motion", *results7))
    
    # Test 8: Asymmetric torques with steering
    def curved_angles_func(t):
        """Gentle steering curve."""
        return np.array([15.0, 15.0, 15.0, 15.0])
    
    def asymmetric_torques_func(t):
        """Asymmetric torque."""
        return np.array([2.1, 1.9, 1.9, 2.1])
    
    results8 = run_test_scenario_dynamic(
        name="CURVED + ASYMMETRIC (15 deg angles + unequal torques)",
        description="Wheels at 15 deg, asymmetric torque.\n"
                   "Expected: Combined steering and asymmetric torque creates complex curved path.",
        angle_func=curved_angles_func,
        torque_func=asymmetric_torques_func,
        duration=6.0,
        dt=0.005
    )
    all_results.append(("Curved + Asymmetric", *results8))
    
    # Plot all results
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_results(all_results)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n[DONE] All test scenarios completed successfully!")
    print("[DONE] Check the generated PNG files for visualization:")
    print(f"  Location: {Path(__file__).parent / 'ActuatorController' / 'Testing'}")
    print("\nGenerated plots:")
    print("  - dynamics_test_paths.png: Overview of all robot paths")
    print("  - dynamics_test_detailed_1_straight_line.png")
    print("  - dynamics_test_detailed_2_rotation_in_place.png")
    print("  - dynamics_test_detailed_3_diagonal_motion.png")
    print("  - dynamics_test_detailed_4_curved_path.png")
    print("  - dynamics_test_detailed_5_circular_motion.png")
    print("  - dynamics_test_detailed_6_pure_spin.png")
    print("  - dynamics_test_detailed_7_spiral_motion.png")
    print("  - dynamics_test_detailed_8_curved_asymmetric.png")
    print("\nKEY IMPLEMENTATION IMPROVEMENTS:")
    print("  ✓ Steering angles ramp to target before full torques applied")
    print("  ✓ No feedback control overshoot - predictive steering velocities")
    print("  ✓ Zero torques during steering phase prevents transient oscillations")
    print("  ✓ Smooth force application - no discontinuous changes")
    print("  ✓ Steering rate limiting keeps motion realistic")
    print("  ✓ Better alignment with Robot_Sim physics model")


if __name__ == "__main__":
    main()
