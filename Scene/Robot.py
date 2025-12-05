from typing import Any, Dict

class Robot:
    def __init__(self, robot_json_object: Dict[str, Any]) -> None:
        self.length: float = robot_json_object["Dimensions"]["Length"]
        self.width: float = robot_json_object["Dimensions"]["Width"]
        self.mass: float = robot_json_object["Mass"]
        self.inertia: float = robot_json_object["Inertia"]
        '''The moment of inertia of the robot in kg*m^2.'''
        self.max_wheel_rotation_speed: float = robot_json_object["Max Wheel Rotation Speed"]
        '''The maximum rotation speed of the wheels in radians per second.'''
        self.max_wheel_acceleration: float = robot_json_object["Max Wheel Acceleration"]
        '''The maximum acceleration of the wheels in radians per second squared.'''
        self.max_wheel_torque: float = robot_json_object["Max Wheel Torque"]
        '''The maximum torque of the wheels in Newton-meters.'''
        self.max_steering_speed: float = robot_json_object["Max Steering Speed"]
        '''The maximum steering speed for the Swerve drive in radians per second.'''
        self.max_steering_acceleration: float = robot_json_object["Max Steering Acceleration"]
        '''The maximum steering acceleration for the Swerve drive in radians per second squared.'''
        self.max_steering_torque: float = robot_json_object["Max Steering Torque"]
        '''The maximum steering torque for the Swerve drive in Newton-meters.'''