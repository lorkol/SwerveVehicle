
import numpy as np
from Scene.Robot import Robot
from ActuatorController.ActuatorController import ActuatorController
from PathController.Types import State_Vector, Control_Vector

class Robot_Sim:
    def __init__(self, robot: Robot):
        self.state = None
        self.robot: Robot = robot
        self.actuator_controller: ActuatorController = ActuatorController(robot)
        # TODO: Get from parameters file
        self.dt = 0.1  # Simulation timestep
        
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
        
        Wheel angles and velocities are updated from control inputs.
        
        Args:
            state: Current state [x, y, theta, vx, vy, omega, d1, d2, d3, d4]
            control_input: Control inputs [tau1, tau2, tau3, tau4, v_d1, v_d2, v_d3, v_d4]
            
        Returns:
            Updated state after dt seconds
        """
        # Unpack current state
        x, y, theta, vx, vy, omega, d1, d2, d3, d4 = state
        
        # Get wheel angles and torques from control input
        wheel_angles = np.array([d1, d2, d3, d4])
        wheel_torques = np.array(control_input[:4])
        
        # Calculate accelerations from actuator controller
        a_x, a_y, alpha = self.actuator_controller.get_accels(wheel_angles, wheel_torques)
        
        # Update positions using kinematic equations with acceleration terms
        # Position change = velocity*dt + 0.5*acceleration*dt²
        x_new = x + vx * self.dt + 0.5 * a_x * self.dt**2
        y_new = y + vy * self.dt + 0.5 * a_y * self.dt**2
        theta_new = theta + omega * self.dt + 0.5 * alpha * self.dt**2
        
        # Update velocities using v_new = v_old + acceleration*dt
        vx_new = vx + a_x * self.dt
        vy_new = vy + a_y * self.dt
        omega_new = omega + alpha * self.dt
        
        # Update wheel angles using wheel velocity commands
        # d_new = d_old + v_d * dt
        wheel_velocities = np.array(control_input[4:8])
        d1_new = d1 + wheel_velocities[0] * self.dt
        d2_new = d2 + wheel_velocities[1] * self.dt
        d3_new = d3 + wheel_velocities[2] * self.dt
        d4_new = d4 + wheel_velocities[3] * self.dt
        
        # Construct and return new state
        new_state: State_Vector = np.array([x_new, y_new, theta_new, vx_new, vy_new, omega_new, d1_new, d2_new, d3_new, d4_new])
        
        return new_state