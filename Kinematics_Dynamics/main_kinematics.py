import numpy as np
import matplotlib.pyplot as plt

from Misc import add_gaussian_noise
from SwerveRobot import SwerveRobot

if __name__ == "__main__":
    swerve_robot: SwerveRobot = SwerveRobot(0.5, 0.5, 0.05)
    dt = 0.01
    total_time =10

    # Create the time vector
    time = np.arange(0, total_time + dt, dt)
    num_steps = len(time)

    # Define the noise parameters
    angle_noise_std = 0.  # Standard deviation for angle noise in degrees
    speed_noise_std = 5.  # Standard deviation for speed noise in rad/s

    wheel_speeds = np.full((4,num_steps), 10)
    wheel_angles = np.full((4, num_steps), np.linspace(0, 360, num_steps))

    # Add Gaussian noise
    wheel_speeds = add_gaussian_noise(wheel_speeds, 0, speed_noise_std)
    wheel_angles = add_gaussian_noise(wheel_angles, 0, angle_noise_std)

    # --- 4. Main Simulation Loop (Replicates the Simulink diagram) ---
    # Initialize state variables
    x, y, theta = 0.0, 0.0, 0.0

    # Lists to store the history for plotting
    x_history, y_history, theta_history = [], [], []
    vx_history, vy_history, w_history = [], [], []

    print("Running simulation...")

    for i in range(num_steps):
        # Get current wheel angles and speeds from the input data
        current_angles = wheel_angles[:, i]
        current_speeds = wheel_speeds[:, i]

        # --- Replicate the Matrix Math and Pseudo-Inverse ---
        # Step 1: Create the A matrix from the current angles
        A = swerve_robot.create_A_Matrix(current_angles[0], current_angles[1], current_angles[2], current_angles[3])
        # Step 2: Compute the pseudo-inverse of A
        A_inv = np.linalg.pinv(A)

        # Step 3: Compute the scaled wheel speeds vector
        speeds_vector = swerve_robot.r * current_speeds

        # Step 4: Perform the matrix multiplication to get the robot's velocity
        # The @ operator is a cleaner way to do matrix multiplication in Python
        robot_velocity_vector = A_inv @ speeds_vector

        vx, vy, w = robot_velocity_vector
        #Drop noise due to calculation in program inaccuracies
        if abs(w) < 0.0000001:
            w = 0

        # Store velocities for plotting
        vx_history.append(vx)
        vy_history.append(vy)
        w_history.append(w)

        # --- Replicate the Integrators ---
        # Integrate to find the new position and orientation
        # This is a simple Euler integration
        x += vx * dt
        y += vy * dt
        theta += w * dt

        # Store position and orientation for plotting
        x_history.append(x)
        y_history.append(y)
        theta_history.append(theta)

    print("Simulation finished. Generating plots...")

    # --- 5. Plot the Results (Replicates the Scope and XY Graph blocks) ---

    # Plot 1: Robot's path on an XY Graph
    plt.figure(figsize=(8, 8))
    plt.plot(x_history, y_history)
    plt.title("Robot Path (X vs Y)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.axis('equal')  # Ensure the scale is equal for a proper visual representation
    plt.show()

    # Plot 2: Velocities and Orientation over time (Replicates the Scopes)
    plt.figure(figsize=(12, 6))

    # Subplot for velocities
    plt.subplot(2, 1, 1)
    plt.plot(time, vx_history, label='Vx (m/s)')
    plt.plot(time, vy_history, label='Vy (m/s)')
    plt.plot(time, w_history, label='Omega (rad/s)')
    plt.title("Robot Velocities and Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity / Angular Velocity")
    plt.legend()
    plt.grid(True)

    # Subplot for orientation
    plt.subplot(2, 1, 2)
    plt.plot(time, theta_history, label='Orientation (rad)')
    plt.title("Robot Orientation")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()