import numpy as np

from ActuatorController.ActuatorController import ActuatorController
from PathController.MPC.Types import Control_Vector, State_Vector, STATE_SIZE, CONTROL_SIZE, FULL_SIZE, FullTraj



#x_(k+1) = x_k + v_x_k * dt
#v_x_(k+1) = v_x_k + a_x_k * dt # NOTE: a_x_k = f_1(u_k(1-4), del_k(1-4))--------B
#y_(k+1) = y_k + v_y_k * dt
#v_y_(k+1) = v_y_k + a_y_k * dt # NOTE: a_y_k = f_2(u_k(1-4), del_k(1-4))---------B
#theta_(k+1) = theta_k + omega_k * dt
#omega_(k+1) = omega_k + alpha_k * dt # NOTE: alpha_k = f_3(u_k(1-4), del_k(1-4)) -----B
#del_1_(k+1) = del_1_k + u_4_k * dt # NOTE: u_4_k = steering rate command for wheel 1
#del_2_(k+1) = del_2_k + u_5_k * dt # NOTE: u_5_k = steering rate command for wheel 2
#del_3_(k+1) = del_3_k + u_6_k * dt # NOTE: u_6_k = steering rate command for wheel 3
#del_4_(k+1) = del_4_k + u_7_k * dt # NOTE: u_7_k = steering rate command for wheel 4

#TODO: Use the parameters from Parameters.json
N: int = 1000
DT = 0.001


def _get_accels(actuator_controller: ActuatorController, controls: Control_Vector, state: State_Vector) -> np.ndarray:
    """Helper function to get accelerations from the ActuatorController given controls and current state."""
    wheel_angles = state[6:STATE_SIZE]  # delta_1 to delta_4
    wheel_torques = controls[0:4]  # tau_1 to tau_4
    accels = actuator_controller.get_accels(wheel_angles, wheel_torques)
    return accels

def create_b(actuator_controller: ActuatorController, x_bar: FullTraj):
    """Creates the equality constraint vector -g(Lecture slides) = b(qp convention) of the qp in the SQP."""
    dt = DT
    x_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((STATE_SIZE,1))  # Initial pose

    # ensure x_bar is a 1D numpy array
    x_bar = np.asarray(x_bar).reshape(-1)

    g = np.zeros((N*STATE_SIZE, 1))

    # initial condition (as column vector)
    g[0:STATE_SIZE, :] = x_bar[0:STATE_SIZE].reshape((STATE_SIZE,1)) - x_0

    for i in range(0, N-1):
        x_k: State_Vector = np.array([
            x_bar[i*FULL_SIZE],
            x_bar[i*FULL_SIZE + 1],
            x_bar[i*FULL_SIZE + 2],
            x_bar[i*FULL_SIZE + 3],
            x_bar[i*FULL_SIZE + 4],
            x_bar[i*FULL_SIZE + 5],
            x_bar[i*FULL_SIZE + 6],
            x_bar[i*FULL_SIZE + 7],
            x_bar[i*FULL_SIZE + 8],
            x_bar[i*FULL_SIZE + 9],
        ]).reshape((STATE_SIZE,1))
        u_k: Control_Vector = np.array([
            x_bar[i*FULL_SIZE + 10],
            x_bar[i*FULL_SIZE + 11],
            x_bar[i*FULL_SIZE + 12],
            x_bar[i*FULL_SIZE + 13],
            x_bar[i*FULL_SIZE + 14],
            x_bar[i*FULL_SIZE + 15],
            x_bar[i*FULL_SIZE + 16],
            x_bar[i*FULL_SIZE + 17],
        ]).reshape((CONTROL_SIZE,1))

        x_k_next: State_Vector = np.array([
            x_bar[(i+1)*FULL_SIZE],
            x_bar[(i+1)*FULL_SIZE + 1],
            x_bar[(i+1)*FULL_SIZE + 2],
            x_bar[(i+1)*FULL_SIZE + 3],
            x_bar[(i+1)*FULL_SIZE + 4],
            x_bar[(i+1)*FULL_SIZE + 5],
            x_bar[(i+1)*FULL_SIZE + 6],
            x_bar[(i+1)*FULL_SIZE + 7],
            x_bar[(i+1)*FULL_SIZE + 8],
            x_bar[(i+1)*FULL_SIZE + 9],
        ]).reshape((STATE_SIZE,1))
        
        A = np.array([
            [1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])      
        
        accels: np.ndarray = _get_accels(actuator_controller, u_k, x_k)
        B = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, dt, 0, 0, 0],
            [0, 0, 0, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, dt]
        ])
        
        C = np.ndarray([[0],
                        [dt*accels[0]],
                        [0], 
                        [dt*accels[1]], 
                        [0], 
                        [dt*accels[2]],
                        [0],
                        [0],
                        [0],
                        [0]]) # type: ignore
        
        g[(i+1)*STATE_SIZE:(i+2)*STATE_SIZE, :] = A@x_k + B@u_k + C - x_k_next
    return -g

# TODO: Go over this function again to verify correctness
def create_A(actuator_controller: ActuatorController, x_bar: FullTraj):
    """
    Gradient of g with respect to x_bar = Jacobian matrix G(Slides) = A(qp convention) for the QP in SQP.
    
    Structure of G (block tridiagonal):
    [   I,  0,  0, ...,  0  ]  (g_0 depends on x_0 only)
    [   A,  B, -I, ...,  0  ]  (g_1 depends on x_0, u_0, x_1)
    [   0,  0,  A, B, -I, ...]  (g_i depends on x_(i-1), u_(i-1), x_i)
    [  ...                    ]
    [   0, ...,  A,  B, -I  ]  (g_N depends on x_(N-1), u_(N-1), x_N)
    
    where:
    - A = ∂g_i/∂x_(i-1) (STATE_SIZE x STATE_SIZE)
    - B = ∂g_i/∂u_(i-1) (STATE_SIZE x CONTROL_SIZE)
    - I = identity matrix (STATE_SIZE x STATE_SIZE)
    """
    dt = DT
    
    # G has shape (N*STATE_SIZE, N*FULL_SIZE)
    G = np.zeros((N * STATE_SIZE, N * FULL_SIZE))
    
    # First row: g_0 = x_0 - x_0_init, depends only on x_0
    # ∂g_0/∂x_0 = I, ∂g_0/∂(everything else) = 0
    G[0:STATE_SIZE, 0:STATE_SIZE] = np.eye(STATE_SIZE)
    
    # Subsequent rows: g_i = A*x_(i-1) + B*u_(i-1) + C - x_i
    for i in range(1, N):
        x_bar_flat = np.asarray(x_bar).reshape(-1)
        
        # Extract state and control at step i-1
        x_k: State_Vector = np.array([
            x_bar_flat[(i-1)*FULL_SIZE + j] for j in range(STATE_SIZE)
        ]).reshape((STATE_SIZE, 1))
        
        u_k: Control_Vector = np.array([
            x_bar_flat[(i-1)*FULL_SIZE + STATE_SIZE + j] for j in range(CONTROL_SIZE)
        ]).reshape((CONTROL_SIZE, 1))
        
        # A matrix (∂g_i/∂x_(i-1))
        A = np.array([
            [1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Place A in G: row i*STATE_SIZE, column (i-1)*FULL_SIZE to (i-1)*FULL_SIZE + STATE_SIZE
        G[i*STATE_SIZE:(i+1)*STATE_SIZE, (i-1)*FULL_SIZE:(i-1)*FULL_SIZE + STATE_SIZE] = A
        
        # B matrix (∂g_i/∂u_(i-1))
        # For swerve drive, accelerations depend on controls and current steering angles
        accels = _get_accels(actuator_controller, u_k, x_k)
        
        B = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, dt, 0, 0, 0],  # ∂(delta_1_(k+1))/∂v_delta_1 = dt
            [0, 0, 0, 0, 0, dt, 0, 0],  # ∂(delta_2_(k+1))/∂v_delta_2 = dt
            [0, 0, 0, 0, 0, 0, dt, 0],  # ∂(delta_3_(k+1))/∂v_delta_3 = dt
            [0, 0, 0, 0, 0, 0, 0, dt]   # ∂(delta_4_(k+1))/∂v_delta_4 = dt
        ])
        jacobian = actuator_controller.get_accels_jacobian(x_k[6:STATE_SIZE].flatten(), u_k[0:4].flatten())

        B[0, 0:4] = dt * jacobian[0, 0:4]  # ∂(a_x)/∂(tau_1..tau_4)
        B[3, 0:4] = dt * jacobian[1, 0:4]  # ∂(a_y)/∂(tau_1..tau_4)
        B[5, 0:4] = dt * jacobian[2, 0:4]  # ∂(alpha)/∂(tau_1..tau_4)
        
        # Place B in G: row i*STATE_SIZE, column (i-1)*FULL_SIZE + STATE_SIZE
        G[i*STATE_SIZE:(i+1)*STATE_SIZE, (i-1)*FULL_SIZE + STATE_SIZE:(i-1)*FULL_SIZE + FULL_SIZE] = B
        
        # -I matrix (∂g_i/∂x_i)
        # Place -I in G: row i*STATE_SIZE, column i*FULL_SIZE to i*FULL_SIZE + STATE_SIZE
        G[i*STATE_SIZE:(i+1)*STATE_SIZE, i*FULL_SIZE:i*FULL_SIZE + STATE_SIZE] = -np.eye(STATE_SIZE)
    
    return G
    