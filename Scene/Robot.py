from typing import Any, Dict, Optional
import random

class Robot:
    def __init__(self, robot_json_object: Dict[str, Any], noise_params: Optional[Dict[str, Any]] = None) -> None:
        self.length: float = robot_json_object["Dimensions"]["Length"]
        self.width: float = robot_json_object["Dimensions"]["Width"]
        self.mass: float = robot_json_object["Mass"]
        self.wheel_radius: float = robot_json_object["Wheel Radius"]
        self.inertia: float = robot_json_object["Inertia"]
        '''The moment of inertia of the robot in kg*m^2.'''
        self.max_wheel_rotation_speed: float = robot_json_object["Max Wheel Rotation Speed"]
        '''The maximum rotation speed of the wheels in radians per second.'''        
        self.max_wheel_torque: float = robot_json_object["Max Wheel Torque"]
        '''The maximum torque of the wheels in Newton-meters.'''
        self.max_steering_speed: float = robot_json_object["Max Steering Speed"]
        '''The maximum steering speed for the Swerve drive in radians per second.'''
        self.max_steering_acceleration: float = robot_json_object["Max Steering Acceleration"]
        '''The maximum steering acceleration for the Swerve drive in radians per second squared.'''
        self.max_steering_torque: float = robot_json_object["Max Steering Torque"]
        '''The maximum steering torque for the Swerve drive in Newton-meters.'''
        if noise_params is not None:
            if noise_params["Enable"]:
                print("Applying parameter uncertainties to robot:")
                # self.mass += random.uniform(-noise_params["mass_uncertainty"], noise_params["mass_uncertainty"]) * self.mass
                # self.inertia += random.uniform(-noise_params["inertia_uncertainty"], noise_params["inertia_uncertainty"]) * self.inertia
                # self.wheel_radius += random.uniform(-noise_params["wheel_radius_uncertainty"], noise_params["wheel_radius_uncertainty"]) * self.wheel_radius
                self.mass += noise_params["mass_uncertainty"] * self.mass
                self.inertia += noise_params["inertia_uncertainty"] * self.inertia
                self.wheel_radius += noise_params["wheel_radius_uncertainty"] * self.wheel_radius