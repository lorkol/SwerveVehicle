from queue import Full
import numpy as np

from PathController.Types import FULL_SIZE, STATE_SIZE, CONTROL_SIZE, FullTraj


#TODO: Use the parameters from Parameters.json
N: int = 1000


Q = np.zeros((STATE_SIZE, STATE_SIZE))
# TODO: Set appropriate weights in Q
R = np.zeros((CONTROL_SIZE, CONTROL_SIZE))
# TODO: Set appropriate weights in R

# P is block diagonal matrix: [[Q, 0], [0, R]]
# Shape: (FULL_SIZE, FULL_SIZE) = (18, 18)
P = np.block([
    [Q, np.zeros((STATE_SIZE, CONTROL_SIZE))],
    [np.zeros((CONTROL_SIZE, STATE_SIZE)), R]
])

#Making a long P for N times, just stacking P diagonally
P_N = np.zeros((FULL_SIZE*N, FULL_SIZE*N))
for i in range(N):
    P_N[i*FULL_SIZE:(i+1)*FULL_SIZE, i*FULL_SIZE:(i+1)*FULL_SIZE] = P


def J(x_bar: FullTraj, desired_path: FullTraj) -> float:
    """
    Compute the quadratic cost function for the MPC.
    
    J = 0.5 * (x_bar - desired_path)^T * P_N * (x_bar - desired_path)
    
    Args:
        x_bar: Current trajectory (N*FULL_SIZE,)
        desired_path: Desired trajectory (N*FULL_SIZE,)
        
    Returns:
        Scalar cost value
    """
    error = np.asarray(x_bar).reshape(-1) - np.asarray(desired_path).reshape(-1)
    cost = 0.5 * error @ P_N @ error
    return float(cost)

def grad_J(x_bar: FullTraj, desired_path: FullTraj) -> np.ndarray:
    """
    Compute the gradient of the cost function J with respect to x_bar.
    
    grad_J = P_N * (x_bar - desired_path)
    
    Args:
        x_bar: Current trajectory (N*FULL_SIZE,)
        desired_path: Desired trajectory (N*FULL_SIZE,)
        
    Returns:
        Gradient vector of shape (N*FULL_SIZE, 1)
    """
    error = np.asarray(x_bar).reshape(-1) - np.asarray(desired_path).reshape(-1)
    gradient = P_N @ error
    return gradient.reshape((-1, 1))

def hessian_J() -> np.ndarray:
    """
    Compute the Hessian of the cost function J.
    
    Since J is quadratic, the Hessian is constant and equals P_N.
    
    Returns:
        Hessian matrix of shape (N*FULL_SIZE, N*FULL_SIZE)
    """
    return P_N