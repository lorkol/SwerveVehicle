import time
import math
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


from ActuatorController.ActuatorController import ActuatorController

from PathController.LocalPlanners.LocalPlanner import LocalPlanner, LocalPlannerTypes
from PathController.LocalPlanners.PathReference import SimpleReferenceGenerator
from PathController.LocalPlanners.PurePursuit import PurePursuitController
from PathController.Robot_Sim import Robot_Sim
from PathController.MPPI.MPPI_Controller import MPPIController
from PathController.Controller import Controller, ControllerTypes
from PathController.LQR_Controller import LQRController
from PathController.MRAC_Controller import MRACController

from ObstacleDetection.ObstacleDetector import StaticObstacleChecker

from PathController.SMC_Controller import SMCController
from Scene.JsonManager import load_json
from Scene.Map import Map
from Scene.Robot import Robot

from Types import PathType, ConvexShape, State2D

from PathPlanning.Planners import Planner, smooth_path
from PathPlanning.RRT_StarPlanner import RRTStarPlanner
from PathPlanning.AStarPlanner import AStarPlanner
# from PathPlanning.HybridAStarPlanner import HybridAStarPlanner
from PathPlanning.DStarPlanner import DStarPlanner
from typing import Any, Callable, Dict, List, Optional, Tuple

from Uncertainties.uncertainty import add_force_uncertainty, add_state_estimation_uncertainty


class Simulation:      
    """Test Planning and Controlling."""
    DEBUG = True  # Set to False to remove all debug prints
    
    def __init__(self, config_path: str = "Scene/Configuration.json", params_path: str = "Scene/Parameters.json"):
        """Initialize controller tester."""
        self.config: Dict[str, Any] = load_json(config_path)
        self.params: Dict[str, Any] = load_json(params_path)
        
        self.init_scene()
        
        # Create obstacle checker
        self.obstacle_checker = StaticObstacleChecker(robot=self.robot_est, obstacles=self.map_obj.obstacles, map_limits=self.world_bounds, use_parallelization=False,
                                                      segment_num_samples=self.params["Obstacle Detection"]["segment_num_samples"],
                                                      collision_clearance=self.params["Obstacle Detection"]["collision_clearance"])
        
        # Create actuator controller
        self.actuator_controller_est = ActuatorController(self.robot_est)
        self.actuator_controller_true = ActuatorController(self.robot_true)

        # Define start and goal        
        self.start: State2D = np.array(self.params["Problem Statement"]["Start"])  # (x, y, theta)
        self.goal: State2D = np.array(self.params["Problem Statement"]["Goal"])  # (x, y, theta)
        
        self.planner: Planner = self.get_planner()

        self.dt: float = self.params["Control"]["dt"]  # Default dt if not specified
        
        self.controller: Controller = None # type: ignore # Will be created after path is planned
    
    def init_scene(self):
        noise_params: Dict[str, Any] = self.params["Noise"]
        self.state_uncertainty: Dict[str, Any] = noise_params["State Estimation Uncertainty"]
        # Create robot and map
        self.robot_est = Robot(self.config["Robot"])
        """The Estimation we have of our robot"""
        self.robot_true = Robot(self.config["Robot"], noise_params["Parameter Uncertainty"])
        '''True robot - with possible uncertainties'''
        print(f"mass difference: {self.robot_true.mass - self.robot_est.mass}, inertia difference: {self.robot_true.inertia - self.robot_est.inertia}, wheel radius difference: {self.robot_true.wheel_radius - self.robot_est.wheel_radius}")
        self.map_obj = Map(self.config["Map"])
        self.world_bounds = ((0, self.map_obj.length), (0, self.map_obj.width))
        
    def get_planner(self) -> Planner:
        """Get planner based on parameters."""
        planner_config = self.params["Path Planning"]
        planner_type = planner_config["Planner"]["type"]
        
        print(f"Creating planner: {planner_type}")
        
        if planner_type == "AStarPlanner":
            params = planner_config["AStarPlanner"]
            return AStarPlanner(grid_resolution=params["grid_resolution"], angle_resolution=params["angle_resolution"], obstacle_checker=self.obstacle_checker, world_bounds=self.world_bounds)
        elif planner_type == "RRTStarPlanner":
            params = planner_config["RRTStarPlanner"]
            return RRTStarPlanner(obstacle_checker=self.obstacle_checker, world_bounds=self.world_bounds, step_size=params["step_size"], goal_sample_rate=params["goal_sample_rate"], rewire_radius_factor=params["rewire_radius_factor"], timeout=params["timeout"])
        elif planner_type == "HybridAStarPlanner":
            #Not working currently
            # params = planner_config["HybridAStarPlanner"]
            # return HybridAStarPlanner(grid_resolution=params["grid_resolution"], angle_bins=params["angle_bins"], obstacle_checker=self.obstacle_checker, world_bounds=self.world_bounds)
            raise NotImplementedError("HybridAStarPlanner is not yet implemented in this tester.")
        elif planner_type == "DStarPlanner":
            #Not working currently
            # params = planner_config["DStarPlanner"]
            # return DStarPlanner(grid_resolution=params["grid_resolution"], angle_resolution=params["angle_resolution"], obstacle_checker=self.obstacle_checker, world_bounds=self.world_bounds)
            raise NotImplementedError("DStarPlanner is not yet implemented in this tester.")
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
    
    def plan_path(self) -> Optional[PathType]:
        """Plan path using configured planner."""
        planner = self.get_planner()
        print(f"\nPlanning path from {self.start} to {self.goal}...")
        path = planner.plan(self.start, self.goal)
        
        if path is None:
            print("❌ No path found!")
            return None
        path = smooth_path(path, self.obstacle_checker, downsample_factor=1)
        print(f"[OK] Path found with {len(path)} waypoints")
        return path
    
    def _get_local_planner_reference_state_method(self, path) -> Callable:
        try:
            controller_params: Dict[str, Any] = self.params["Control"]["Cascading Controllers"] # TODO: Currently only using Cascading Controllers, add a way to choose
            local_planner_params: Dict[str, Any] = controller_params["LocalPlanner"]
            local_planner_str = local_planner_params["type"]
            local_planner_type = LocalPlannerTypes(local_planner_str)
            if local_planner_type == LocalPlannerTypes.PurePursuit:
                local_planner_params = local_planner_params[local_planner_type.value]
                lookahead = local_planner_params["lookahead"]
                v_desired = local_planner_params["v_desired"]
                pure_pursuit = PurePursuitController(robot_controller=self.actuator_controller_est, path_points=path, lookahead=lookahead, v_desired=v_desired, dt=self.dt)
                return pure_pursuit.get_reference_state
            else:
                raise NotImplementedError # TODO: Implement other traj generators
        except KeyError:
            raise NotImplementedError # TODO: Make a default traj generator
        
    def get_controller(self, path = None) -> Controller:
        """Get controller based on parameters."""
        controller_params: Dict[str, Any] = self.params["Control"]
        controller_params = controller_params["Cascading Controllers"] # TODO: Currently only using this, add a way to choose
        controller_type_str = controller_params["Controller"]["type"]
        try:
            controller_type = ControllerTypes(controller_type_str)
        except (ValueError, KeyError):
            print(f"Warning: Unknown controller type '{controller_type_str}', using LQRController as default.")
            controller_type = ControllerTypes.LQR
        controller_params = controller_params[controller_type.value]
        # Create robot simulator
        dt: float = self.dt
        
        if controller_type == ControllerTypes.LQR:
            # Use a reduced lookahead for the LQR to keep references closer
            # This helps avoid large overshoots when approaching the final goal.
            Q = controller_params["Q"]
            R = controller_params["R"]
            controller: Controller = LQRController(robot_controller=self.actuator_controller_est, get_reference_method=self._get_local_planner_reference_state_method(path), dt=dt, Q=Q, R=R)
        elif controller_type == ControllerTypes.MRAC:
            gamma: float = controller_params["gamma"]
            kp: float = controller_params["kp"]
            kv: float = controller_params["kv"]
            alpha_min: float = controller_params["alpha_min"]
            alpha_max: float = controller_params["alpha_max"]
            controller: Controller = MRACController(robot_controller=self.actuator_controller_est, get_reference_method=self._get_local_planner_reference_state_method(path), dt=dt, gamma=gamma, kp=kp, kv=kv, alpha_min=alpha_min, alpha_max=alpha_max)
        elif controller_type == ControllerTypes.SMC:
            k_gains: np.ndarray = controller_params["k_gains"]
            lambda_gains: np.ndarray = controller_params["lambda_gains"]
            boundary_layer: float = controller_params["boundary_layer"]
            controller: Controller = SMCController(robot_controller=self.actuator_controller_est, get_reference_method=self._get_local_planner_reference_state_method(path), k_gains=k_gains, lambda_gains=lambda_gains, boundary_layer=boundary_layer)
        else: #MPPI or others
            #controller: Controller = MPPIController(desired_traj=path, robot_sim=robot_sim, collision_check_method=self.obstacle_checker.is_collision, N_Horizon=controller_params["N_Horizon"], lambda_=controller_params["Lambda"], myu=controller_params["myu"], K=controller_params["K"])
            raise NotImplementedError("simulation not implemented in this tester.")
        
        return controller
    
    def simulate_controller(self, ref_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate controller following the given path and reference trajectory."""
        # For LQR/MRAC, ref_traj is just the path as (3, N) array, so nothing else needed

        assert self.controller is not None, "Controller not initialized. Call get_controller() first."
                
        robot_sim = Robot_Sim(self.actuator_controller_true, self.robot_true, dt=self.dt)

        # Get stabilization radius from parameters (default 1.0)
        stabilization_radius = self.params["Control"]["StabilizationRadius"]
        stabilization_window = 100
        
        # Initial state
        initial_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot_sim.set_state(initial_state)
        
        # Simulation loop
        executed_states = [initial_state.copy()]
        executed_controls = []
        reference_states = []  # Store reference states from PathFollower
        current_state = initial_state.copy()
        
        if self.DEBUG:
            print(f"\n[DEBUG] Initial state: {initial_state[:3]}")
            print(f"[DEBUG] Reference path shape: {ref_traj.shape}")
            print(f"[DEBUG] Ref path first 3 elements (x,y,theta): {ref_traj[:3, 0]}")
        
        dt_sim: float = self.dt
        max_time = 3  # seconds (real wall-clock time)
        step = 0
        start_time = time.time()
        print(f"\nSimulating controller (real-time timeout: {max_time}s, dt: {dt_sim})...")
        while True:
            # Get reference state from path follower for debugging
            ref_state = self.controller.get_reference_state(current_state[:3])
            reference_states.append(ref_state.copy())
            # Compute control from controller
            if self.state_uncertainty["Enable"]:
                state_noise = add_state_estimation_uncertainty(self.state_uncertainty["Position Noise StdDev"], self.state_uncertainty["Orientation Noise StdDev"], 
                                                               self.state_uncertainty["Linear Velocity Noise StdDev"], self.state_uncertainty["Angular Velocity Noise StdDev"])
                measured_state = current_state + state_noise
            else:
                measured_state = current_state.copy()

            control_input = self.controller.get_command(measured_state)
            executed_controls.append(control_input.copy())
            # Propagate state using robot simulator
            noise_params = self.params["Noise"]
            if noise_params["Dynamic Disturbance"]["Enable"]:
                disturbance_noise = add_force_uncertainty(noise_params["Dynamic Disturbance"]["Max x Force"],
                                                          noise_params["Dynamic Disturbance"]["Max y Force"], 
                                                          noise_params["Dynamic Disturbance"]["Max Torque"])
            else:
                disturbance_noise = np.zeros(3)
            new_state = robot_sim.propagate(current_state, control_input, noise=disturbance_noise)
            executed_states.append(new_state.copy())

            # Print wheel angles (delta1, delta2, delta3, delta4)
            # wheel_angles = new_state[6:10]
            # print(f"Step {step+1}: wheel angles = {[f'{angle:.6f}' for angle in wheel_angles]}")
            current_state = new_state

            # Check stabilization
            if self.state_uncertainty["Enable"]:
                noise = add_state_estimation_uncertainty(self.state_uncertainty["Position Noise StdDev"], self.state_uncertainty["Orientation Noise StdDev"], 
                                                        self.state_uncertainty["Linear Velocity Noise StdDev"], self.state_uncertainty["Angular Velocity Noise StdDev"])
            else:
                noise = 0.0
            # --- New: Windowed stabilization check ---
            stabilized_by_window = False
            if len(executed_states) > stabilization_window:
                recent_states = np.array(executed_states[-stabilization_window:])
                dists = np.linalg.norm(recent_states[:, :2] - np.array(self.goal[:2]), axis=1)
                if np.all(dists < stabilization_radius):
                    stabilized_by_window = True
            if self.controller.is_stabilized(current_state + noise) or stabilized_by_window:
                print(f"[OK] Controller stabilized at step {step+1} (sim t={dt_sim*(step+1):.2f}s, real t={time.time()-start_time:.2f}s)")
                if stabilized_by_window:
                    print(f"[INFO] Stabilization detected: last {stabilization_window} states all within {stabilization_radius}m of goal.")
                break
            # Check real wall-clock timeout
            if (time.time() - start_time) > max_time:
                print(f"[TIMEOUT] Simulation stopped after {max_time} seconds real time at step {step+1} (sim t={dt_sim*(step+1):.2f}s)")
                break
            if (step + 1) % 1000 == 0:
                speed = np.sqrt(current_state[3]**2 + current_state[4]**2)
                print(f"  Step {step + 1}: Position ({current_state[0]:.2f}, {current_state[1]:.2f}), Speed {speed:.3f} m/s, Heading {math.degrees(current_state[2]):.2f}°, torques: {control_input[:4]}")
                if control_input[0] < 1e-3 and control_input[1] < 1e-3 and control_input[2] < 1e-3 and control_input[3] < 1e-3:
                    controller.get_command(current_state, debug=True) #type: ignore
            step += 1
        
        executed_states = np.array(executed_states)
        executed_controls = np.array(executed_controls)
        reference_states = np.array(reference_states)
        
        print(f"[OK] Simulation completed: {len(executed_states)} states")
        return executed_states, executed_controls, reference_states
    
    def calculate_cross_track_error(self, robot_history, path_points):
        """
        Calculates the perpendicular distance from the robot to the path.
        """
        path_points = np.array(path_points)
        cross_track_errors = []
        
        for robot_pos in robot_history:
            rx, ry = robot_pos[0], robot_pos[1]
            
            # Find distance to every segment and pick the smallest
            min_dist = float('inf')
            
            for i in range(len(path_points) - 1):
                p1 = path_points[i][:2] if len(path_points[i]) > 2 else path_points[i]  # Handle both 2D and 3D points
                p2 = path_points[i+1][:2] if len(path_points[i+1]) > 2 else path_points[i+1]
                
                # Distance from point (rx, ry) to line segment p1-p2
                # Standard "Point to Line Segment" formula
                p1_p2 = p2 - p1
                len_sq = np.dot(p1_p2, p1_p2)
                
                if len_sq == 0:
                    dist = np.linalg.norm(np.array([rx, ry]) - p1)
                else:
                    t = np.dot(np.array([rx, ry]) - p1, p1_p2) / len_sq
                    t = np.clip(t, 0, 1)  # Clamp to segment
                    projection = p1 + t * p1_p2
                    dist = np.linalg.norm(np.array([rx, ry]) - projection)
                
                if dist < min_dist:
                    min_dist = dist
                    
            cross_track_errors.append(min_dist)
            
        return np.array(cross_track_errors)
    
    def calculate_path_errors(self, robot_history, path_points):
        """
        Calculate x, y, and theta errors relative to the closest point on the path.
        Returns separate error arrays for x, y, and theta.
        """
        path_points = np.array(path_points)
        error_x = []
        error_y = []
        error_theta = []
        
        for robot_pos in robot_history:
            rx, ry = robot_pos[0], robot_pos[1]
            r_theta = robot_pos[2]
            
            # Find closest point on path
            min_dist = float('inf')
            closest_path_point = None
            closest_idx = 0
            
            for i in range(len(path_points) - 1):
                p1 = path_points[i][:2] if len(path_points[i]) > 2 else path_points[i]
                p2 = path_points[i+1][:2] if len(path_points[i+1]) > 2 else path_points[i+1]
                
                # Find projection on segment
                p1_p2 = p2 - p1
                len_sq = np.dot(p1_p2, p1_p2)
                
                if len_sq == 0:
                    projection = p1
                    t = 0
                else:
                    t = np.dot(np.array([rx, ry]) - p1, p1_p2) / len_sq
                    t = np.clip(t, 0, 1)
                    projection = p1 + t * p1_p2
                
                dist = np.linalg.norm(np.array([rx, ry]) - projection)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_path_point = path_points[i] if t < 0.5 else path_points[i+1]
                    closest_idx = i if t < 0.5 else i+1
            
            # Calculate errors
            px = closest_path_point[0] # type: ignore
            py = closest_path_point[1] # type: ignore
            p_theta = closest_path_point[2] if len(closest_path_point) > 2 else 0 # type: ignore
            
            error_x.append(rx - px)
            error_y.append(ry - py)
            
            # Handle theta wrapping
            theta_diff = r_theta - p_theta
            while theta_diff > np.pi:
                theta_diff -= 2 * np.pi
            while theta_diff < -np.pi:
                theta_diff += 2 * np.pi
            error_theta.append(theta_diff)
        
        return np.array(error_x), np.array(error_y), np.array(error_theta)
    
    def visualize(self, path: PathType, executed_states: np.ndarray, ref_traj: np.ndarray, reference_states: np.ndarray = None):
        """Visualize planning and control results."""
        print(f"\nVisualizing: {len(executed_states)} executed states")
        if len(executed_states) > 0:
            print(f"  First state: ({executed_states[0][0]:.2f}, {executed_states[0][1]:.2f})")
            print(f"  Last state: ({executed_states[-1][0]:.2f}, {executed_states[-1][1]:.2f})")
        if reference_states is not None:
            print(f"  Reference states collected: {len(reference_states)}")
        
        # Create a single large figure for the trajectory/map
        fig, ax1 = plt.subplots(figsize=(20, 12))

        (x_min, x_max), (y_min, y_max) = self.world_bounds

        ax1.set_xlim(x_min - 1, x_max + 1)
        ax1.set_ylim(y_min - 1, y_max + 1)
        ax1.set_aspect('equal')
        ax1.set_title('Controller Execution Trajectory', fontsize=14, fontweight='bold')

        # Map boundary
        boundary = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.1
        )
        ax1.add_patch(boundary)
        
        # Obstacles
        for i, obstacle in enumerate(self.map_obj.obstacles):
            if obstacle.shape == ConvexShape.Circle:
                circle = patches.Circle(
                    (obstacle.center[0], obstacle.center[1]), obstacle.radius, # type: ignore
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.4,
                    label='Obstacles' if i == 0 else ''
                )
                ax1.add_patch(circle)
            elif obstacle.shape == ConvexShape.Polygon:
                polygon = patches.Polygon(
                    obstacle.points, linewidth=2, edgecolor='red', facecolor='red', alpha=0.4,
                    label='Obstacles' if i == 0 else ''
                )
                ax1.add_patch(polygon)
        
        # Reference trajectory
        # Reference trajectory (now just the path for LQR/MRAC)
        ref_x = ref_traj[0, :]
        ref_y = ref_traj[1, :]
        ax1.plot(ref_x, ref_y, 'g--', linewidth=1, label='Reference Path', alpha=0.7)
        
        # Executed trajectory
        if len(executed_states) > 0:
            exec_x = [state[0] for state in executed_states]
            exec_y = [state[1] for state in executed_states]
            ax1.plot(exec_x, exec_y, 'b-', linewidth=2, label='Executed Trajectory')
            ax1.scatter(exec_x, exec_y, s=10, c='blue', alpha=0.5, zorder=5)
        
        # PathFollower reference states (what the controller is actually tracking)
        if reference_states is not None and len(reference_states) > 0:
            pf_ref_x = reference_states[:, 0]
            pf_ref_y = reference_states[:, 1]
            ax1.plot(pf_ref_x, pf_ref_y, 'orange', linewidth=2, linestyle=':', label='PathFollower Ref', zorder=6)
            # Show theta direction arrows at intervals
            arrow_step = max(1, len(reference_states) // 15)
            for i in range(0, len(reference_states), arrow_step):
                x, y, theta = reference_states[i, 0], reference_states[i, 1], reference_states[i, 2]
                dx = 0.8 * np.cos(theta)
                dy = 0.8 * np.sin(theta)
                ax1.arrow(x, y, dx, dy, head_width=0.4, head_length=0.2, fc='orange', ec='darkorange', alpha=0.7, zorder=6)
            
            # Draw robot rectangles at regular intervals
            robot_length = self.robot_true.length
            robot_width = self.robot_true.width
            num_total = 20  # total rectangles to draw
            num_pre90 = 10   # at least 10 before 90% progress
            # Compute cumulative path length
            path_points = np.array(path)
            seg_lengths = np.linalg.norm(path_points[1:, :2] - path_points[:-1, :2], axis=1)
            cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = cum_lengths[-1]

            def get_progress(x, y):
                # Find closest segment and fraction along path
                dists = np.linalg.norm(path_points[:, :2] - np.array([x, y]), axis=1)
                idx = np.argmin(dists)
                if idx == len(cum_lengths) - 1:
                    return 1.0
                seg_start = path_points[idx, :2]
                seg_end = path_points[min(idx+1, len(path_points)-1), :2]
                seg_vec = seg_end - seg_start
                if np.linalg.norm(seg_vec) < 1e-6:
                    frac = 0.0
                else:
                    frac = np.dot([x, y] - seg_start, seg_vec) / np.dot(seg_vec, seg_vec)
                    frac = np.clip(frac, 0, 1)
                length_along = cum_lengths[idx] + frac * (cum_lengths[min(idx+1, len(cum_lengths)-1)] - cum_lengths[idx])
                return length_along / total_length if total_length > 0 else 0.0

            # Select indices for rectangles
            indices_pre90 = []
            indices_post90 = []
            for i in range(len(executed_states)):
                x, y = executed_states[i][0], executed_states[i][1]
                progress = get_progress(x, y)
                if progress < 0.9:
                    indices_pre90.append(i)
                else:
                    indices_post90.append(i)
            # Sample at least 10 before 90%, rest after
            if len(indices_pre90) > 0:
                step_pre = max(1, len(indices_pre90) // num_pre90)
                chosen_pre = indices_pre90[::step_pre][:num_pre90]
            else:
                chosen_pre = []
            num_post = num_total - len(chosen_pre)
            if len(indices_post90) > 0 and num_post > 0:
                step_post = max(1, len(indices_post90) // num_post)
                chosen_post = indices_post90[::step_post][:num_post]
            else:
                chosen_post = []
            chosen_indices = sorted(set(chosen_pre + chosen_post))
            for i in chosen_indices:
                state = executed_states[i]
                x, y, theta = state[0], state[1], state[2]
                rect = patches.Rectangle(
                    (-robot_length/2, -robot_width/2), robot_length, robot_width,
                    linewidth=1.5, edgecolor='darkblue', facecolor='cyan', alpha=0.3
                )
                t = Affine2D().rotate_around(0, 0, theta) + Affine2D().translate(x, y)
                t += ax1.transData
                rect.set_transform(t)
                ax1.add_patch(rect)
                # Draw an arrow for theta direction from the center of the rectangle
                arrow_length = robot_length * 0.7
                dx = arrow_length * np.cos(theta)
                dy = arrow_length * np.sin(theta)
                ax1.arrow(x, y, dx, dy, head_width=robot_width*0.4, head_length=robot_length*0.25, fc='navy', ec='navy', alpha=0.8, zorder=8)
        
        # Start and goal
        ax1.scatter(*self.start[:2], s=200, c='green', marker='o', edgecolors='darkgreen', linewidth=2, label='Start', zorder=10)
        ax1.scatter(*self.goal[:2], s=200, c='red', marker='s', edgecolors='darkred', linewidth=2, label='Goal', zorder=10)
        
        ax1.set_xlabel('X (meters)', fontsize=12)
        ax1.set_ylabel('Y (meters)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Calculate path errors (x, y, theta relative to closest path point)
        error_x, error_y, error_theta = self.calculate_path_errors(executed_states, path)
        cross_track_errors = self.calculate_cross_track_error(executed_states, path)
        if self.DEBUG:
            print(f"\n[DEBUG] Path errors calculated: {len(error_x)} points")

        # Print path tracking error statistics
        print(f"\n--- Path Tracking Error Statistics ---")
        print(f"  X Error - Mean: {np.mean(np.abs(error_x)):.4f}m, Max: {np.max(np.abs(error_x)):.4f}m, RMS: {np.sqrt(np.mean(error_x**2)):.4f}m")
        print(f"  Y Error - Mean: {np.mean(np.abs(error_y)):.4f}m, Max: {np.max(np.abs(error_y)):.4f}m, RMS: {np.sqrt(np.mean(error_y**2)):.4f}m")
        print(f"  Theta Error - Mean: {np.mean(np.abs(error_theta)):.4f}rad, Max: {np.max(np.abs(error_theta)):.4f}rad, RMS: {np.sqrt(np.mean(error_theta**2)):.4f}rad")
        print(f"  Cross-Track Error - Mean: {np.mean(cross_track_errors):.4f}m, Max: {np.max(cross_track_errors):.4f}m, RMS: {np.sqrt(np.mean(cross_track_errors**2)):.4f}m")

        plt.tight_layout()
        # Save figure instead of blocking on show()
        plt.savefig('controller_test_result.png', dpi=150, bbox_inches='tight')
        print("Figure saved to controller_test_result.png")
        plt.show()

        # Show velocity and error plots in a second popup
        self.plot_velocities_and_errors(executed_states, error_x, error_y, error_theta, cross_track_errors)
    
    def plot_velocities_and_errors(self, executed_states: np.ndarray, error_x, error_y, error_theta, cross_track_errors):
        """
        Plots velocity and tracking error graphs in a single popup window.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if executed_states.shape[1] < 6:
            print("[WARN] executed_states does not have enough columns for velocities. Skipping velocity plots.")
            return

        vx = executed_states[:, 3]
        vy = executed_states[:, 4]
        vtheta = executed_states[:, 5]
        time = np.arange(len(executed_states))

        fig, axs = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle('Velocity and Tracking Error Graphs')

        # Velocity graphs
        axs[0, 0].plot(time, vx, color='b')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Velocity x (m/s)')
        axs[0, 0].set_title('vx over time')

        axs[0, 1].plot(time, vy, color='g')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Velocity y (m/s)')
        axs[0, 1].set_title('vy over time')

        axs[0, 2].plot(time, vtheta, color='r')
        axs[0, 2].set_xlabel('Time Step')
        axs[0, 2].set_ylabel('Velocity theta (rad/s)')
        axs[0, 2].set_title('vtheta over time')

        # Tracking error graphs
        axs[1, 0].plot(time, error_x, 'r-', linewidth=2, label='X Error')
        axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 0].fill_between(time, 0, error_x, alpha=0.3, color='red')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Error (meters)')
        axs[1, 0].set_title('X Position Error vs Path')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        axs[1, 1].plot(time, error_y, 'b-', linewidth=2, label='Y Error')
        axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 1].fill_between(time, 0, error_y, alpha=0.3, color='blue')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Error (meters)')
        axs[1, 1].set_title('Y Position Error vs Path')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()

        axs[1, 2].plot(time, error_theta, 'g-', linewidth=2, label='Theta Error')
        axs[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 2].fill_between(time, 0, error_theta, alpha=0.3, color='green')
        axs[1, 2].set_xlabel('Time Step')
        axs[1, 2].set_ylabel('Error (radians)')
        axs[1, 2].set_title('Theta (Angle) Error vs Path')
        axs[1, 2].grid(True, alpha=0.3)
        axs[1, 2].legend()

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig('velocity_and_tracking_errors.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def generate_trajectory(self, path: PathType) -> np.ndarray:
        """Return the path as a (3, N) array for plotting as the reference trajectory."""
        arr = np.array(path).T  # shape (3, N)
        return arr
    
    def run_single_iteration(self):
        """Run complete controller test."""
        print("=" * 60)
        print("CONTROLLER TEST")
        print("=" * 60)
        
        # Plan path
        path = self.plan_path() # TODO: Can make a path from another source - what's called in get_controller() if oath is None
        if path is None:
            return
        # Generate trajectory
        ref_traj: np.ndarray = self.generate_trajectory(path)
        # Simulate controller
        self.controller = self.get_controller(path)
        executed_states, executed_controls, reference_states = self.simulate_controller(ref_traj)
        # Visualize
        print("\nVisualizing results...")
        self.visualize(path, executed_states, ref_traj, reference_states)
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        
    # TODO: dynamic controller for possible changing path for replanning situations

if __name__ == "__main__":
    try:
        tester = Simulation()
        tester.run_single_iteration()
    except Exception as e:
        print(f"\n✗ Error during controller testing: {e}")
        import traceback
        traceback.print_exc()


