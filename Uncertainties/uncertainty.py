import numpy as np


def add_state_estimation_uncertainty(pos_stddev: float, orient_stddev: float,
                                     lin_vel_stddev: float, ang_vel_stddev: float) -> np.ndarray:
    """TODO: Add docstring.

    Args:
        TODO: describe parameters

    Returns:
        TODO: describe return value
    """
    return np.array([np.random.normal(0, pos_stddev), np.random.normal(0, pos_stddev),
                     np.random.normal(0, orient_stddev), np.random.normal(0, lin_vel_stddev),
                     np.random.normal(0, lin_vel_stddev), np.random.normal(0, ang_vel_stddev), 
                     0.0, 0.0, 0.0, 0.0])
    
def add_force_uncertainty(max_fx: float, max_fy: float, max_torque: float) -> np.ndarray:
    """TODO: Add docstring.

    Args:
        TODO: describe parameters

    Returns:
        TODO: describe return value
    """
    return np.array([np.random.uniform(-max_fx, max_fx), np.random.uniform(-max_fy, max_fy), np.random.uniform(-max_torque, max_torque)])

def create_parameter_uncertainty_multiplier(uncertainty: float) -> float:
    """Create a multiplier for a parameter based on the given uncertainty percentage."""
    return np.clip(0.01, np.random.uniform(1 - uncertainty, 1 + uncertainty), np.inf)