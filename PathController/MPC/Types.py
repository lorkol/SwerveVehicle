
import numpy as np
from Types import TypeAlias


State_Vector: TypeAlias = np.ndarray
'''(10,1) Represents the full state vector of the vehicle:\n
[x, y, theta, v_x, v_y, omega, delta_1, delta_2, delta_3, delta_4]'''
STATE_SIZE = 10

Control_Vector: TypeAlias = np.ndarray
'''(8,1) Represents the full control vector of the vehicle:\n
[tau_1, tau_2, tau_3, tau_4, v_delta_1, v_delta_2, v_delta_3, v_delta_4]'''
CONTROL_SIZE = 8

# Type alias for FullState as numpy array
# Elements: [x, y, theta, v_x, v_y, omega, delta_1, delta_2, delta_3, delta_4]
FullState: TypeAlias = np.ndarray
'''(18,1) Represents the full state vector of the vehicle WITH the controls:\n
[x, y, theta, v_x, v_y, omega, delta_1, delta_2, delta_3, delta_4, tau_1, tau_2, tau_3, tau_4, v_delta_1, v_delta_2, v_delta_3, v_delta_4]'''
FULL_SIZE = STATE_SIZE + CONTROL_SIZE # = 18

FullTraj: TypeAlias = np.ndarray
'''Represents the full trajectory over the horizon as a long numpy array of shape (N*len(FullState),)'''