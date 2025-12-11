from math import exp
from typing import TypeAlias, Callable
import numpy as np
from Types import State
from PathController.Types import CONTROL_SIZE, STATE_SIZE, ControlSequence, Trajectory, State_Vector, Control_Vector
from Robot_Sim import Robot_Sim

# TODO: make the desired trajectory from the path planner according to the N_Horizon and dt of the robot simulation

Q = np.zeros((STATE_SIZE, STATE_SIZE))
Q[0, 0] = 1.0  # Weight for x position error
Q[1, 1] = 1.0  # Weight for y position error
Q[2, 2] = 0.5  # Weight for theta error
# TODO: Tweak weights and put in parameters file
R = np.zeros((CONTROL_SIZE, CONTROL_SIZE))
# TODO: Set appropriate weights in R

# P is block diagonal matrix: [[Q, 0], [0, R]]
# Shape: (FULL_SIZE, FULL_SIZE) = (18, 18)
P = np.block([
    [Q, np.zeros((STATE_SIZE, CONTROL_SIZE))],
    [np.zeros((CONTROL_SIZE, STATE_SIZE)), R]
])


def propagate(robot_sim: Robot_Sim, collision_check_method: Callable[[State], bool], state_0: State_Vector, control_sequence: ControlSequence) -> Trajectory:
    """Propagate the robot state over a control sequence."""
    N_Horizon = control_sequence.shape[1]
    traj: Trajectory = np.zeros((STATE_SIZE, N_Horizon))
    current_state: State_Vector = state_0.copy()
    for t in range(N_Horizon):
        traj[:, t] = current_state
        # Extract 1D control vector for this timestep
        control_input: Control_Vector = control_sequence[:, t]
        current_state = robot_sim.propagate(current_state, control_input)
        if collision_check_method(tuple(current_state[:3])):  # Check collision with (x, y, theta)
            raise Exception("Collision detected during trajectory propagation.")
    return traj


class MPPIController:
    def __init__(self, desired_traj: Trajectory, robot_sim: Robot_Sim, collision_check_method: Callable[[State], bool]) -> None:       
        # TODO: Import from parameters file
        self.N = 500  # number of timesteps
        self.N_Horizon = 10
        self.alpha: float = 1.
        self.sigma: float = 1.
        self.Lambda: float = 0.1
        self.myu: float = 2.
        self.K: int = 50
        self.desired_traj: Trajectory = desired_traj
        self.u_bar: ControlSequence = np.reshape(np.random.multivariate_normal(np.zeros(CONTROL_SIZE*self.N_Horizon), np.eye(CONTROL_SIZE*self.N_Horizon)),
                                                 [CONTROL_SIZE, self.N_Horizon])
        self.robot_sim: Robot_Sim = robot_sim
        self.collision_check_method: Callable[[State], bool] = collision_check_method

    def l_cost(self, u_bar: ControlSequence, state_0: State_Vector = np.zeros(STATE_SIZE)) -> float:
        """Compute cost for a given control sequence."""
        try:
            traj: Trajectory = propagate(self.robot_sim, self.collision_check_method, state_0, u_bar)
        except Exception as e:
            return float('inf')  # High cost for invalid trajectories (e.g., collisions)
        cost: float = 0.
        for i in range(self.N_Horizon):
            err = self.desired_traj[:, i] - traj[:, i]
            cost += float((err).T @ Q @ (err))
        return cost

    def u_bar_update(self, u_bar: ControlSequence, state_0: State_Vector) -> ControlSequence:
        """Update control sequence using MPPI."""
        exp_sum_nom = np.reshape(np.zeros(CONTROL_SIZE*self.N_Horizon), [CONTROL_SIZE, self.N_Horizon])
        exp_sum_denom: float = 0.
        # TODO: Find a way to incorporate a timeout if no random sample is found within the timeout, reduce the horizon the algoritm is running on for the iteration. for very narrow passages
        for i in range(self.K):
            epsilon_k = np.reshape(
                np.random.multivariate_normal(np.zeros(CONTROL_SIZE*self.N_Horizon), np.eye(CONTROL_SIZE*self.N_Horizon)), 
                [CONTROL_SIZE, self.N_Horizon]
            )
            u_candidate = u_bar + self.myu * epsilon_k
            cost_val = self.l_cost(u_candidate, state_0)
            if cost_val == float('inf'):
                i -= 1
                continue  # Skip invalid trajectories while still guaranteeing having enough valid ones
            weight = exp(-(1/self.Lambda) * cost_val) / self.sigma
            exp_sum_nom += weight * epsilon_k
            exp_sum_denom += exp(-(1/self.Lambda) * cost_val)
        
        if exp_sum_denom == 0:
            return u_bar
        
        return u_bar - self.alpha * (-self.Lambda * exp_sum_nom) / (self.myu * exp_sum_denom)

    def get_command(self, state: State_Vector) -> Control_Vector:
        """Get the next control command."""
        self.u_bar = self.u_bar_update(np.column_stack((self.u_bar[:, 1:], np.zeros((CONTROL_SIZE, 1)))), state)  # Stacking 0's into the last command
        return self.u_bar[:, 0]
