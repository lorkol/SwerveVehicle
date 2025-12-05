from Scene.JsonManager import JsonManager


class Robot:
    def __init__(self, robot_file: str) -> None:
        self._robot_json = JsonManager(robot_file)
        self.length: float = self._robot_json.read_param_value("Dimensions/Length")
        self.width: float = self._robot_json.read_param_value("Dimensions/Width")
        self.mass: float = self._robot_json.read_param_value("Mass")
        self.inertia: float = self._robot_json.read_param_value("Inertia")
        self.max_wheel_rotation_speed: float = self._robot_json.read_param_value("Max Wheel Rotation Speed")
        '''The maximum rotation speed of the wheels in radians per second.'''
        self.max_wheel_acceleration: float = self._robot_json.read_param_value("Max Wheel Acceleration")
        '''The maximum acceleration of the wheels in radians per second squared.'''
        self.max_wheel_torque: float = self._robot_json.read_param_value("Max Wheel Torque")
        '''The maximum torque of the wheels in Newton-meters.'''
        self.max_steering_speed: float = self._robot_json.read_param_value("Max Steering Speed")
        '''The maximum steering speed for the Swerve drive in radians per second.'''
        self.max_steering_acceleration: float = self._robot_json.read_param_value("Max Steering Acceleration")
        '''The maximum steering acceleration for the Swerve drive in radians per second squared.'''
        self.max_steering_torque: float = self._robot_json.read_param_value("Max Steering Torque")
        '''The maximum steering torque for the Swerve drive in Newton-meters.'''