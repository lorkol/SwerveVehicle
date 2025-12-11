
import numpy as np
from Types import TypeAlias


State_Vector: TypeAlias = np.ndarray
'''Represents the state of the swerve drive robot (10, 1):\n
    - x, y: Position in global frame\n
    - theta: Orientation angle in global frame\n
    - vx_R, vy_R: Velocities in robot frame\n
    - v_theta: Angular velocity\n
    - delta1, delta2, delta3, delta4: Steering angles of each wheel'''
STATE_SIZE = 10

Control_Vector: TypeAlias = np.ndarray
'''Represents control inputs for the swerve drive robot: (8,1)\n
    - Tau1, Tau2, Tau3, Tau4: Torques applied at each wheel\n
    - Delta_dot1, Delta_dot2, Delta_dot3, Delta_dot4: Steering angular velocities for each wheel
'''
CONTROL_SIZE = 8

# Type alias for FullState as numpy array
# Elements: [x, y, theta, v_x, v_y, omega, delta_1, delta_2, delta_3, delta_4]
FullState: TypeAlias = np.ndarray
'''(18,1) Represents the full state vector of the vehicle WITH the controls:\n
[x, y, theta, v_x, v_y, omega, delta_1, delta_2, delta_3, delta_4, tau_1, tau_2, tau_3, tau_4, v_delta_1, v_delta_2, v_delta_3, v_delta_4]'''
FULL_SIZE = STATE_SIZE + CONTROL_SIZE # = 18


ControlSequence: TypeAlias = np.ndarray  # Shape: (8, N)
'''A sequence of control inputs over N timesteps = (8, N)'''

Trajectory: TypeAlias = np.ndarray  # Shape: (10, N)
'''A sequence of states over N timesteps = (10, N)'''