import numpy as np
from PathController.Types import Control_Vector, State_Vector


# TODO: Use parameters from Config.json
Max_Wheel_Rotation_Speed = 10.0
Max_Wheel_Torque = 15.0
Max_Steering_Speed = 30.0
Wheel_Radius = 0.1
Map_Length = 100.0  # meters
Map_Width = 50.0   # meters

max_velocity = Max_Wheel_Rotation_Speed * Wheel_Radius

def state_velocity_constraint(state: State_Vector) -> float:
    state_x_vel = state[1]
    state_y_vel = state[3]
    velocity_magnitude = (state_x_vel**2 + state_y_vel**2)**0.5
    return velocity_magnitude - max_velocity

def _torque_limit_constraint(inputs: Control_Vector) -> np.ndarray:
    wheel_torques = inputs[0:4]
    return np.array([abs(wheel_torques[0] - Max_Wheel_Torque),
                     abs(wheel_torques[1] - Max_Wheel_Torque), 
                     abs(wheel_torques[2] - Max_Wheel_Torque), 
                     abs(wheel_torques[3] - Max_Wheel_Torque)])
    
def _steering_speed_constraint(inputs: Control_Vector) -> np.ndarray:
    steering_speeds = inputs[4:8]
    return np.array([abs(steering_speeds[0] - Max_Steering_Speed),
                     abs(steering_speeds[1] - Max_Steering_Speed),
                     abs(steering_speeds[2] - Max_Steering_Speed),
                     abs(steering_speeds[3] - Max_Steering_Speed)])
    
def input_constraints(inputs: Control_Vector) -> np.ndarray:
    torque_constraints = _torque_limit_constraint(inputs)
    steering_constraints = _steering_speed_constraint(inputs)
    return np.concatenate((torque_constraints, steering_constraints), axis=0)

def state_position_constraints(state: State_Vector) -> np.ndarray:
    """Ensures the vehicle stays within the map boundaries.
    
    Args:
        state (State_Vector): The current state vector of the vehicle.
        
    Returns:
        np.ndarray: An array containing the position constraints.
                    The first element should be non-negative if within the left boundary,
                    the second element should be non-negative if within the right boundary,
                    the third element should be non-negative if within the bottom boundary,
                    and the fourth element should be non-negative if within the top boundary.
    """
    x_pos = state[0]
    y_pos = state[2]
    
    left_boundary = -x_pos  # should be <= 0
    right_boundary = x_pos - Map_Length  # should be <= 0
    bottom_boundary = -y_pos  # should be <= 0
    top_boundary = y_pos - Map_Width  # should be <= 0
    
    return np.array([left_boundary, right_boundary, bottom_boundary, top_boundary])
