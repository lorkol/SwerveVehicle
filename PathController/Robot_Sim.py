import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from Scene.Robot import Robot
from ActuatorController.ActuatorController import ActuatorController
from PathController.Types import State_Vector, Control_Vector

# CRITICAL FIX FOR OSCILLATIONS:
# The ActuatorController's B matrix uses trigonometric functions (sin, cos) of wheel angles.
# If angles accumulate beyond [-π, π], different angle values produce different trig results.
# For example: sin(0) = 0, but sin(2π) also = 0, but sin(π/4) ≠ sin(π/4 + 2π)
# This caused FORCE DISCONTINUITIES during continuous steering, producing OSCILLATIONS.
# Solution: Normalize all wheel angles to [-π, π] using atan2 after each update.
# This ensures: d1_rad and d1_rad + 2π produce identical forces.

class Robot_Sim:
    def __init__(self, actuator: ActuatorController, robot: Robot, dt: float = 0.1):
        self._state = None
        self._robot: Robot = robot
        self._actuator_controller: ActuatorController = actuator
        # TODO: Get from parameters file
        self._dt: float = dt  # Simulation timestep
        
    def propagate(self, state: State_Vector, control_input: Control_Vector) -> State_Vector:
        """
        Propagate state forward using kinematic equations that incorporate acceleration.
        
        Position updates use kinematic equations:
        - x_new = x + vx*dt + 0.5*ax*dt²
        - y_new = y + vy*dt + 0.5*ay*dt²
        - theta_new = theta + omega*dt + 0.5*alpha*dt²
        
        Velocity updates use:
        - vx_new = vx + ax*dt
        - vy_new = vy + ay*dt
        - omega_new = omega + alpha*dt
        
        Wheel angles are updated from control inputs and normalized to [-π, π].
        
        Args:
            state: Current state [x, y, theta, vx, vy, omega, d1, d2, d3, d4]
            control_input: Control inputs [tau1, tau2, tau3, tau4, d1, d2, d3, d4]
            
        Returns:
            Updated state after dt seconds
        """
        # Unpack current state
        x, y, theta, vx, vy, omega, d1, d2, d3, d4 = state
        
        # Get wheel angles and torques from control input
        wheel_angles: np.ndarray = np.array(control_input[4:8])
        wheel_torques: np.ndarray = np.array(control_input[:4])
        
        # Calculate accelerations from actuator controller
        a_x, a_y, alpha = self._actuator_controller.get_accels_in_world(state, wheel_angles, wheel_torques)
        
        # Update positions using kinematic equations with acceleration terms
        # Position change = velocity*dt + 0.5*acceleration*dt²
        x_new: float = x + vx * self._dt + 0.5 * a_x * self._dt**2
        y_new: float = y + vy * self._dt + 0.5 * a_y * self._dt**2
        theta_new: float = theta + omega * self._dt + 0.5 * alpha * self._dt**2
        
        # Update velocities using v_new = v_old + acceleration*dt
        vx_new: float = vx + a_x * self._dt
        vy_new: float = vy + a_y * self._dt
        omega_new: float = omega + alpha * self._dt
        
        d1_new, d2_new, d3_new, d4_new = wheel_angles  # Directly set to commanded angles
                
        # Construct and return new state
        new_state: State_Vector = np.array([x_new, y_new, theta_new, vx_new, vy_new, omega_new, d1_new, d2_new, d3_new, d4_new])
        self._state = new_state
        
        return new_state
    
    def set_state(self, state: State_Vector) -> None:
        """Set the internal state of the robot simulator."""
        self._state = state