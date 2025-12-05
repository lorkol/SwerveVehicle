from typing import Tuple
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# --- INPUT PARAMETERS: Modify these values to run different scenarios ---
TIME_TOTAL = 10.0  # Total simulation time (s)
TIME_STEPS = 2000  # Number of simulation steps

# Command Inputs: Torque applied to all four wheels (FR, FL, RL, RR)
BASE_TORQUE = 1.  # N*m
# Example: Apply a slight torque asymmetry to induce rotation
WHEEL_TORQUES = np.array([1. * BASE_TORQUE, 1. * BASE_TORQUE, 1. * BASE_TORQUE, 1. * BASE_TORQUE])

# Command Inputs: Steering angle for all four wheels (FR, FL, RL, RR)
# Example: Steer slightly to the right to see a pronounced curve
ANGLE_DEGREE = 0.0  # Try 0.0 for straight or 15.0 for a curve
WHEEL_ANGLES = np.radians(np.array([ANGLE_DEGREE, ANGLE_DEGREE, ANGLE_DEGREE, ANGLE_DEGREE]))

# --- ROBOT CONSTANTS ---
m = 50.0  # kg (Mass)
I = 4.0  # kg*m^2 (Moment of Inertia about Z-axis)
L = 0.5  # m (Half-length)
W = 0.4  # m (Half-width)
r = 0.05  # m (Wheel radius)

M_MATRIX = np.array([[m, 0., 0.], [0., m, 0.0], [0., 0., I]])
# State Vector: [x, y, theta, vx_R, vy_R, v_theta]
X0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
TIME_POINTS = np.linspace(0, TIME_TOTAL, TIME_STEPS)


# --- DYNAMIC MODEL FUNCTIONS ---

def build_B(angles: np.ndarray, l: float, w: float) -> np.ndarray:
    """
    Builds the control input matrix B (3x4) which maps wheel forces (tau/r)
    in the Robot Frame to body forces [Fx_R, Fy_R, Tau_theta].
    """
    delta1, delta2, delta3, delta4 = angles

    L_pos, W_pos = l, w

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


def swerve_dynamics_ode(X: np.ndarray, t: float, M_mat: np.ndarray, wheel_angles: np.ndarray, wheel_torques: np.ndarray,
                        l: float, w: float, r: float) -> np.ndarray:
    """
    The function for the ODE solver (odeint). It calculates the derivatives dX/dt.
    X: [x, y, theta, vx_R, vy_R, v_theta]
    dX/dt: [vx_G, vy_G, v_theta, ax_R, ay_R, a_theta]
    """
    # Unpack current state
    x, y, theta, vx_R, vy_R, v_theta = X

    # --- 1. Calculate accelerations in the Robot Frame ---
    B_mat = build_B(wheel_angles, l, w)
    F_R = (B_mat / r).dot(wheel_torques)

    # Calculate accelerations in the Robot Frame: a = M^{-1} * F
    accels_R = np.linalg.solve(M_mat, F_R)
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


def swerve_inverse_kinematics_analytical(accels_R: np.ndarray, M_mat: np.ndarray,
                                          l: float, w: float, r: float) -> Tuple[np.ndarray, np.ndarray]:
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
    F_R = M_mat.dot(accels_R)  # F = M * a
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
    B_mat = build_B(wheel_angles, l, w)
    
    # Step 4: Solve for wheel torques to produce both linear AND angular acceleration
    # We have 3 equations (Fx, Fy, Tau) and 4 unknowns (tau1, tau2, tau3, tau4)
    # Use least squares to find the minimum-norm solution
    # This will automatically make wheels on one side faster for rotation
    
    wheel_forces = np.linalg.lstsq(B_mat, F_R, rcond=None)[0]
    wheel_torques = r * wheel_forces
    
    return wheel_angles, wheel_torques



# --- MAIN EXECUTION ---

# Example: Inverse kinematics
print("=== INVERSE KINEMATICS EXAMPLE ===")
desired_accels = np.array([2.0, 0.5, 0.1])  # ax_R, ay_R, a_theta
print(f"Desired accelerations (Robot Frame): {desired_accels}")
print()

# RECOMMENDED: Analytical solution (fast, no optimization needed)
print("✓ Analytical Solution (Fast):")
angles_analytical, torques_analytical = swerve_inverse_kinematics_analytical(
    desired_accels, M_MATRIX, L, W, r
)
print(f"  Steering angles: {np.degrees(angles_analytical)}°")
print(f"  Wheel torques: {torques_analytical}")
print()


# --- FORWARD DYNAMICS SIMULATION ---

# Print initial acceleration for verification
B_initial = build_B(WHEEL_ANGLES, L, W)
accels_initial = np.linalg.solve(M_MATRIX, (B_initial / r).dot(WHEEL_TORQUES))
print(f"Initial Robot Frame Accelerations: [ax_R, ay_R, a_theta] = {accels_initial} (m/s^2, rad/s^2)")
print("-" * 50)

# Solve the ODE (Integrate the dynamics over time)
solution = odeint(
    swerve_dynamics_ode,
    X0,
    TIME_POINTS,
    args=(M_MATRIX, WHEEL_ANGLES, WHEEL_TORQUES, L, W, r)
)

# Separate the results
x_G = solution[:, 0]
y_G = solution[:, 1]
theta = solution[:, 2]
vx_R = solution[:, 3]
vy_R = solution[:, 4]
v_theta = solution[:, 5]

# Calculate Global Frame Velocities for the velocity plot
vx_G = vx_R * np.cos(theta) - vy_R * np.sin(theta)
vy_G = vx_R * np.sin(theta) + vy_R * np.cos(theta)

# The accelerations are the derivatives calculated in the ODE,
# which is the instantaneous value of the last three components of dX/dt.
# We run the ODE function *without* changing the state X to extract the
# instantaneous derivatives at every point.

# Create an array to store the accelerations [ax_R, ay_R, a_theta]
accels_R_all = np.zeros((TIME_STEPS, 3))
for i, X in enumerate(solution):
    dXdt = swerve_dynamics_ode(X, TIME_POINTS[i], M_MATRIX, WHEEL_ANGLES, WHEEL_TORQUES, L, W, r)
    # The accelerations are the last three elements of dXdt: [ax_R, ay_R, a_theta]
    accels_R_all[i] = dXdt[3:6]

ax_R = accels_R_all[:, 0]
ay_R = accels_R_all[:, 1]
a_theta = accels_R_all[:, 2]

# Transform accelerations from Robot Frame (ax_R, ay_R) to Global Frame (ax_G, ay_G)
# Note: The acceleration transformation is the same as the velocity transformation (V_I = R*V_R)
ax_G = ax_R * np.cos(theta) - ay_R * np.sin(theta)
ay_G = ax_R * np.sin(theta) + ay_R * np.cos(theta)

# --- PLOTTING ---

fig, axs = plt.subplots(3, 2, figsize=(10, 12))
fig.suptitle(f"Swerve Robot Dynamics Simulation (Total Time: {TIME_TOTAL}s, Angle: {ANGLE_DEGREE}°)")

# 1. Plot Path (Global Frame Position)
axs[0][0].plot(x_G, y_G, label='Robot Path')
axs[0][0].set_title('Robot Path (Global Frame: $x_I$ vs $y_I$)')
axs[0][0].set_xlabel('$x_I$ Position (m)')
axs[0][0].set_ylabel('$y_I$ Position (m)')
axs[0][0].grid(True)
axs[0][0].set_aspect('equal', adjustable='box')  # Keep the path visually accurate

# 2. Plot Velocities (Global and Rotational)
axs[1][0].plot(TIME_POINTS, vx_G, label='$v_{xI}$ (Global $x$)')
axs[1][0].plot(TIME_POINTS, vy_G, label='$v_{yI}$ (Global $y$)')
axs[1][0].plot(TIME_POINTS, v_theta, label='$v_{\\theta}$ (Angular)')
axs[1][0].set_title('Velocities over Time')
axs[1][0].set_xlabel('Time (s)')
axs[1][0].set_ylabel('Velocity (m/s or rad/s)')
axs[1][0].legend()
axs[1][0].grid(True)

# 3. Plot Accelerations (Global and Rotational)
axs[2][0].plot(TIME_POINTS, ax_G, label='$a_{xI}$ (Global $x$)')
axs[2][0].plot(TIME_POINTS, ay_G, label='$a_{yI}$ (Global $y$)')
axs[2][0].plot(TIME_POINTS, a_theta, label='$a_{\\theta}$ (Angular)')
axs[2][0].set_title('Accelerations over Time (Global Frame $a_{xI}$, $a_{yI}$)')
axs[2][0].set_xlabel('Time (s)')
axs[2][0].set_ylabel('Acceleration ($m/s^2$ or $rad/s^2$)')
axs[2][0].legend()
axs[2][0].grid(True)

# 4. Plot positions (Robot Frame Position)
axs[0][1].plot(TIME_POINTS, x_G, label='Robot x')
axs[0][1].plot(TIME_POINTS, y_G, label='Robot y')
axs[0][1].set_title('Robot Path (Robot Frame: $x_I$ vs $y_I$)')
axs[0][1].set_xlabel('Time (s)')
axs[0][1].set_ylabel('Position')
axs[0][1].grid(True)
axs[0][1].legend()
axs[0][1].set_aspect('equal', adjustable='box')  # Keep the path visually accurate


# 4. Plot Velocities (Robot and Rotational)
axs[1][1].plot(TIME_POINTS, vx_R, label='$v_{xI}$ (Robot $x$)', linewidth=4)
axs[1][1].plot(TIME_POINTS, vy_R, label='$v_{yI}$ (Robot $y$)')
axs[1][1].plot(TIME_POINTS, v_theta, label='$v_{\\theta}$ (Angular)')
axs[1][1].set_title('Velocities over Time')
axs[1][1].set_xlabel('Time (s)')
axs[1][1].set_ylabel('Velocity (m/s or rad/s)')
axs[1][1].legend()
axs[1][1].grid(True)

# 5. Plot Accelerations (In Robot Frame)
axs[2][1].plot(TIME_POINTS, ax_R, label='$a_{xI}$ (Robot $x$)', linewidth=4)
axs[2][1].plot(TIME_POINTS, ay_R, label='$a_{yI}$ (Robot $y$)')
axs[2][1].plot(TIME_POINTS, a_theta, label='$a_{\\theta}$ (Angular)')
axs[2][1].set_title('Accelerations over Time (Robot Frame $a_{xI}$, $a_{yI}$)')
axs[2][1].set_xlabel('Time (s)')
axs[2][1].set_ylabel('Acceleration ($m/s^2$')
axs[2][1].legend()
axs[2][1].grid(True)

plt.tight_layout()
plt.show()