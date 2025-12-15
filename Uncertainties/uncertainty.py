import numpy as np


def add_state_estimation_uncertainty(pos_stddev: float, orient_stddev: float,
                                     lin_vel_stddev: float, ang_vel_stddev: float) -> np.ndarray:
    return np.array([np.random.normal(0, pos_stddev), np.random.normal(0, pos_stddev),
                     np.random.normal(0, orient_stddev), np.random.normal(0, lin_vel_stddev),
                     np.random.normal(0, lin_vel_stddev), np.random.normal(0, ang_vel_stddev), 
                     0.0, 0.0, 0.0, 0.0])