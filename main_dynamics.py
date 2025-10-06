import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- INPUT PARAMETERS: Modify these values to run different scenarios ---
TIME_TOTAL = 20.0  # Total simulation time (s)
TIME_STEPS = 2000  # Number of simulation steps

# Command Inputs: Torque applied to all four wheels (FR, FL, RL, RR)
BASE_TORQUE = 0.1  # N*m
# Example: Apply a slight torque asymmetry to induce rotation
WHEEL_TORQUES = np.array([1. * BASE_TORQUE, 1. * BASE_TORQUE, 1. * BASE_TORQUE, 1. * BASE_TORQUE])

# Command Inputs: Steering angle for all four wheels (FR, FL, RL, RR)
# Example: Steer slightly to the right to see a pronounced curve
ANGLE_DEGREE = 30.0  # Try 0.0 for straight or 15.0 for a curve
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


# --- MAIN EXECUTION ---

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

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle(f"Swerve Robot Dynamics Simulation (Total Time: {TIME_TOTAL}s, Angle: {ANGLE_DEGREE}°)")

# 1. Plot Path (Global Frame Position)
axs[0].plot(x_G, y_G, label='Robot Path')
axs[0].set_title('Robot Path (Global Frame: $x_I$ vs $y_I$)')
axs[0].set_xlabel('$x_I$ Position (m)')
axs[0].set_ylabel('$y_I$ Position (m)')
axs[0].grid(True)
axs[0].set_aspect('equal', adjustable='box')  # Keep the path visually accurate

# 2. Plot Velocities (Global and Rotational)
axs[1].plot(TIME_POINTS, vx_G, label='$v_{xI}$ (Global $x$)')
axs[1].plot(TIME_POINTS, vy_G, label='$v_{yI}$ (Global $y$)')
axs[1].plot(TIME_POINTS, v_theta, label='$v_{\\theta}$ (Angular)')
axs[1].set_title('Velocities over Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Velocity (m/s or rad/s)')
axs[1].legend()
axs[1].grid(True)

# 3. Plot Accelerations (Global and Rotational)
axs[2].plot(TIME_POINTS, ax_G, label='$a_{xI}$ (Global $x$)')
axs[2].plot(TIME_POINTS, ay_G, label='$a_{yI}$ (Global $y$)')
axs[2].plot(TIME_POINTS, a_theta, label='$a_{\\theta}$ (Angular)')
axs[2].set_title('Accelerations over Time (Global Frame $a_{xI}$, $a_{yI}$)')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Acceleration ($m/s^2$ or $rad/s^2$)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()