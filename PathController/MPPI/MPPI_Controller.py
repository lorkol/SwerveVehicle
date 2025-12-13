import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from math import exp
from typing import Callable
import numpy as np
from PathController.Controller import Controller
from Types import State2D
from PathController.Types import CONTROL_SIZE, STATE_SIZE, ControlSequence, Trajectory, State_Vector, Control_Vector
from PathController.Robot_Sim import Robot_Sim

# TODO: make the desired trajectory from the path planner according to the N_Horizon and dt of the robot simulation

Q = np.zeros((STATE_SIZE, STATE_SIZE))
Q[0, 0] = 1.0  # Weight for x position error
Q[1, 1] = 1.0  # Weight for y position error
Q[2, 2] = 0.5  # Weight for theta error
# Only position/angle are weighted, not velocities
R = np.zeros((CONTROL_SIZE, CONTROL_SIZE))
# TODO: Set appropriate weights in R

# P is block diagonal matrix: [[Q, 0], [0, R]]
# Shape: (FULL_SIZE, FULL_SIZE) = (18, 18)
P = np.block([
    [Q, np.zeros((STATE_SIZE, CONTROL_SIZE))],
    [np.zeros((CONTROL_SIZE, STATE_SIZE)), R]
])



class MPPIController(Controller):
    """Model Predictive Path Integral (MPPI) Controller."""
    def __init__(self, desired_traj: Trajectory, robot_sim: Robot_Sim, collision_check_method: Callable[[State2D], bool], N_Horizon: int = 5, alpha: float = 1.0, sigma: float = 1.0, lambda_: float = 0.1, myu: float = 2.0, K: int = 50) -> None:       
        self._N_Horizon = N_Horizon
        self._alpha: float = alpha
        self._sigma: float = sigma
        self._Lambda: float = lambda_
        self._myu: float = myu
        self._K: int = K
        self._desired_traj: Trajectory = desired_traj
        self._u_bar: ControlSequence = np.zeros((CONTROL_SIZE, self._N_Horizon))  # Initialize to zero, not random
        self._robot_sim: Robot_Sim = robot_sim
        self._collision_check_method: Callable[[State2D], bool] = collision_check_method

    def _l_cost(self, u_bar: ControlSequence, state_0: State_Vector = np.zeros(STATE_SIZE)) -> float:
        """Compute cost for a given control sequence."""
        try:
            traj: Trajectory = self._propagate_trajectory(state_0, u_bar)
        except Exception as e:
            return float('inf')  # High cost for invalid trajectories (e.g., collisions)
        cost: float = 0.
        for i in range(self._N_Horizon):
            err = self._desired_traj[:, i] - traj[:, i]
            cost += float((err).T @ Q @ (err))
        return cost

    def _u_bar_update(self, u_bar: ControlSequence, state_0: State_Vector) -> ControlSequence:
        """Update control sequence using MPPI."""
        exp_sum_nom: ControlSequence = np.reshape(np.zeros(CONTROL_SIZE*self._N_Horizon), [CONTROL_SIZE, self._N_Horizon])
        exp_sum_denom: float = 0.
        # TODO: Find a way to incorporate a timeout if no random sample is found within the timeout, reduce the horizon the algoritm is running on for the iteration. for very narrow passages
        for i in range(self._K): # TODO: Consider parallelizing this loop in GPU in the future for speedup
            epsilon_k: ControlSequence = np.reshape(np.random.multivariate_normal(np.zeros(CONTROL_SIZE*self._N_Horizon), np.eye(CONTROL_SIZE*self._N_Horizon)),
                                                    [CONTROL_SIZE, self._N_Horizon])
            u_candidate: ControlSequence = u_bar + self._myu * epsilon_k
            cost_val: float = self._l_cost(u_candidate, state_0)
            # TODO: Consider Not skipping cause then it might never reach K samples, and then just give high cost
            if cost_val == float('inf'):
                i -= 1
                continue  # Skip invalid trajectories while still guaranteeing having enough valid ones
            weight = exp(-(1/self._Lambda) * cost_val) / self._sigma
            exp_sum_nom += weight * epsilon_k
            exp_sum_denom += exp(-(1/self._Lambda) * cost_val)
        
        if exp_sum_denom == 0:
            return u_bar
        
        # TODO: Consider that this avg might have a collision even if all samples were collision free individually
        return u_bar - self._alpha * (-self._Lambda * exp_sum_nom) / (self._myu * exp_sum_denom)

    def _propagate_trajectory(self, state_0: State_Vector, control_sequence: ControlSequence) -> Trajectory:
        """Propagate the robot state over a control sequence."""
        traj: Trajectory = np.zeros((STATE_SIZE, self._N_Horizon))
        current_state: State_Vector = state_0.copy()
        for t in range(self._N_Horizon):
            traj[:, t] = current_state
            # Extract 1D control vector for this timestep
            control_input: Control_Vector = control_sequence[:, t]
            current_state = self._robot_sim.propagate(current_state, control_input)
            if self._collision_check_method(tuple(current_state[:3])):  # Check collision with (x, y, theta)
                self._robot_sim.set_state(state_0)
                raise Exception("Collision detected during trajectory propagation.")
        self._robot_sim.set_state(state_0)
        return traj
    
    def get_command(self, state: State_Vector) -> Control_Vector:
        """Get the next control command."""
        self._u_bar = self._u_bar_update(np.column_stack((self._u_bar[:, 1:], np.zeros((CONTROL_SIZE, 1)))), state)  # Stacking 0's into the last command
        return self._u_bar[:, 0]
