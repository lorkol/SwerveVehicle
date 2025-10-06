import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- INPUT PARAMETERS: Modify these values to run different scenarios ---
TIME_TOTAL = 20.0  # Total simulation time (s)
TIME_STEPS = 200  # Number of simulation steps

# Command Inputs: Torque applied to all four wheels (FR, FL, RL, RR)
BASE_TORQUE = 0.1  # N*m
# Example: Apply a slight torque asymmetry to induce rotation
WHEEL_TORQUES = np.array([1.1 * BASE_TORQUE, BASE_TORQUE, BASE_TORQUE, 1.1 * BASE_TORQUE])

# Command Inputs: Steering angle for all four wheels (FR, FL, RL, RR)
# Example: Steer slightly to the right to see a pronounced curve
ANGLE_DEGREE = 0.0
WHEEL_ANGLES = np.radians(np.array([ANGLE_DEGREE, ANGLE_DEGREE, ANGLE_DEGREE, ANGLE_DEGREE]))

# --- ROBOT CONSTANTS ---
m = 50.0  # kg
I = 4.0  # kg*m^2
L = 0.5  # m (Half-length)
W = 0.4  # m (Half-width)
r = 0.05  # m (Wheel radius)

M_MATRIX = np.array([[m, 0., 0.], [0., m, 0.0], [0., 0., I]])
X0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, vx, vy, v_theta]
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

B = build_B(WHEEL_ANGLES, L, W)
accels: np.ndarray = np.linalg.solve(M_MATRIX, (B / r).dot(WHEEL_TORQUES))

print(accels)