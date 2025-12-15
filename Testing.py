import time
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


from ActuatorController.ActuatorController import ActuatorController

from PathController.PurePursuit import PurePursuitController
from PathController.Robot_Sim import Robot_Sim
from PathController.MPPI.MPPI_Controller import MPPIController
from PathController.Controller import Controller, ControllerTypes, LocalPlanner, LocalPlannerTypes
from PathController.LQR_Controller import LQRController
from PathController.MRAC_Controller import MRACController
from PathController.PathReference import ProjectedPathFollower

from ObstacleDetection.ObstacleDetector import StaticObstacleChecker

from Scene.JsonManager import load_json
from Scene.Map import Map
from Scene.Robot import Robot

from Types import PathType, ConvexShape

from PathPlanning.Planners import Planner
from PathPlanning.RRT_StarPlanner import RRTStarPlanner
from PathPlanning.AStarPlanner import AStarPlanner
from PathPlanning.HybridAStarPlanner import HybridAStarPlanner
from PathPlanning.DStarPlanner import DStarPlanner
from typing import Any, Dict, Optional, Tuple

from Uncertainties.uncertainty import add_state_estimation_uncertainty


class ControllerTester:
    """Test Planning and Controlling."""
    DEBUG = True  # Set to False to remove all debug prints
    
    def __init__(self, config_path: str = "Scene/Configuration.json", params_path: str = "Scene/Parameters.json"):
        """Initialize controller tester."""
        self.config: Dict[str, Any] = load_json(config_path)
        self.params: Dict[str, Any] = load_json(params_path)
        
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
        
        # Create obstacle checker
        self.obstacle_checker = StaticObstacleChecker(robot=self.robot_est, obstacles=self.map_obj.obstacles, map_limits=self.world_bounds, use_parallelization=False)
        
        # Create actuator controller
        self.actuator_controller_est = ActuatorController(self.robot_est)
        self.actuator_controller_true = ActuatorController(self.robot_true)

        # Define start and goal        
        self.start = (5.0, 5.0, 0.0)
        self.goal = (self.map_obj.length - 20.0, self.map_obj.width - 20.0, 0.0)
        self.planner: Planner = self.get_planner()

        
    
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
            params = planner_config["HybridAStarPlanner"]
            return HybridAStarPlanner(grid_resolution=params["grid_resolution"], angle_bins=params["angle_bins"], obstacle_checker=self.obstacle_checker, world_bounds=self.world_bounds)
        elif planner_type == "DStarPlanner":
            params = planner_config["DStarPlanner"]
            return DStarPlanner(grid_resolution=params["grid_resolution"], angle_resolution=params["angle_resolution"], obstacle_checker=self.obstacle_checker, world_bounds=self.world_bounds)
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
        
        print(f"[OK] Path found with {len(path)} waypoints")
        return path
    
    def generate_trajectory(self, path: PathType) -> np.ndarray:
        """Return the path as a (3, N) array for plotting as the reference trajectory."""
        arr = np.array(path).T  # shape (3, N)
        return arr

    def simulate_controller(self, path: PathType, ref_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate MPPI controller following reference trajectory with rolling horizon."""
        # Debug: Check path endpoint
        print(f"\n[DEBUG] PathFollower path check:")
        print(f"  Original goal: ({self.goal[0]:.4f}, {self.goal[1]:.4f})")
        print(f"  Path last point: ({path[-1][0]:.4f}, {path[-1][1]:.4f})")
        
        # Ensure path ends exactly at the intended goal (prevent small shifts)
        try:
            path[-1] = (self.goal[0], self.goal[1], self.goal[2])
        except Exception:
            pass
        # For LQR/MRAC, ref_traj is just the path as (3, N) array, so nothing else needed

        # Create Controller
        controller_params: Dict[str, Any] = self.params["Control"]
        controller_params = controller_params["Cascading Controllers"] # TODO: Currently only using this, add a way to choose
        local_planner_params: Dict[str, Any] = controller_params["LocalPlanner"]
        local_planner_str = local_planner_params["type"]
        controller_type_str = controller_params["Controller"]["type"]
        try:
            controller_type = ControllerTypes(controller_type_str)
            local_planner_type = LocalPlannerTypes(local_planner_str)
        except (ValueError, KeyError):
            print(f"Warning: Unknown controller type '{controller_type_str}', using LQRController or PurePursuitController as default.")
            controller_type = ControllerTypes.LQR
            local_planner_type = LocalPlannerTypes.PurePursuit
        controller_params = controller_params[controller_type.value]
        local_planner_params = local_planner_params[local_planner_type.value]
        # Create robot simulator
        dt: float = controller_params["dt"]
        robot_sim = Robot_Sim(self.actuator_controller_true, self.robot_true, dt=dt)
        if local_planner_type == LocalPlannerTypes.PurePursuit:
            local_planner = PurePursuitController(robot_controller=self.actuator_controller_est, path_points=path,
                                                  lookahead=local_planner_params["lookahead"], v_desired=controller_params["v_desired"], dt=local_planner_params["dt"])
        else:
            raise NotImplementedError("Only PurePursuit local planner is implemented in this tester.")
        
        if controller_type == ControllerTypes.LQR:
            # Use a reduced lookahead for the LQR to keep references closer
            # This helps avoid large overshoots when approaching the final goal.
            v_desired: float = controller_params["v_desired"] # TODO: Check consistency in v_desired with local planner
            dt: float = controller_params["dt"] # TODO: Check consistency in dt with local planner
            Q = controller_params["Q"]
            R = controller_params["R"]
            controller: Controller = LQRController(robot_controller=self.actuator_controller_est, local_planner=local_planner, v_desired=v_desired, dt=dt, Q=Q, R=R)
        elif controller_type == ControllerTypes.MRAC:
            dt: float = controller_params["dt"] # TODO: Check consistency in dt with local planner
            gamma: float = controller_params["gamma"]
            kp: float = controller_params["kp"]
            kv: float = controller_params["kv"]
            alpha_min: float = controller_params["alpha_min"]
            alpha_max: float = controller_params["alpha_max"]
            controller: Controller = MRACController(robot_controller=self.actuator_controller_est, local_planner=local_planner, dt=dt, gamma=gamma, kp=kp, kv=kv, alpha_min=alpha_min, alpha_max=alpha_max)
        else: #MPPI or others
            #controller: Controller = MPPIController(desired_traj=path, robot_sim=robot_sim, collision_check_method=self.obstacle_checker.is_collision, N_Horizon=controller_params["N_Horizon"], lambda_=controller_params["Lambda"], myu=controller_params["myu"], K=controller_params["K"])
            raise NotImplementedError("simulation not implemented in this tester.")
        
        
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
        
        dt_sim = controller_params["dt"]
        max_time = 15.0  # seconds (real wall-clock time)
        step = 0
        start_time = time.time()
        print(f"\nSimulating controller (real-time timeout: {max_time}s, dt: {dt_sim})...")
        while True:
            # Get reference state from path follower for debugging
            ref_state = controller.get_reference_state(current_state[:3])
            reference_states.append(ref_state.copy())
            # Compute control from controller
            if self.state_uncertainty["Enable"]:
                noise = add_state_estimation_uncertainty(self.state_uncertainty["Position Noise StdDev"], self.state_uncertainty["Orientation Noise StdDev"], 
                                                         self.state_uncertainty["Linear Velocity Noise StdDev"], self.state_uncertainty["Angular Velocity Noise StdDev"])
                measured_state = current_state + noise
            else:
                measured_state = current_state.copy()
                
            control_input = controller.get_command(measured_state)
            executed_controls.append(control_input.copy())
            # Propagate state using robot simulator
            current_state = robot_sim.propagate(current_state, control_input)
            executed_states.append(current_state.copy())
            # Check stabilization
            if self.state_uncertainty["Enable"]:
                noise = add_state_estimation_uncertainty(self.state_uncertainty["Position Noise StdDev"], self.state_uncertainty["Orientation Noise StdDev"], 
                                                        self.state_uncertainty["Linear Velocity Noise StdDev"], self.state_uncertainty["Angular Velocity Noise StdDev"])
            else:
                noise = 0.0
            if controller.is_stabilized(current_state + noise):
                print(f"[OK] Controller stabilized at step {step+1} (sim t={dt_sim*(step+1):.2f}s, real t={time.time()-start_time:.2f}s)")
                break
            # Check real wall-clock timeout
            if (time.time() - start_time) > max_time:
                print(f"[TIMEOUT] Simulation stopped after {max_time} seconds real time at step {step+1} (sim t={dt_sim*(step+1):.2f}s)")
                break
            if (step + 1) % 100 == 0:
                speed = np.sqrt(current_state[3]**2 + current_state[4]**2)
                print(f"  Step {step + 1}: Position ({current_state[0]:.2f}, {current_state[1]:.2f}), Speed {speed:.3f} m/s, Heading {math.degrees(current_state[2]):.2f}°, torques: {control_input[:4]}")
                if control_input[0] < 1e-3 and control_input[1] < 1e-3 and control_input[2] < 1e-3 and control_input[3] < 1e-3:
                    controller.get_command(current_state, debug=True)
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
        
        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
        
        (x_min, x_max), (y_min, y_max) = self.world_bounds
        
        # --- Plot 1: Controller Execution (Large, spans 2 columns) ---
        ax1 = fig.add_subplot(gs[0, 0:2])
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
            robot_length = self.robot_est.length
            robot_width = self.robot_est.width
            step_interval = max(1, len(executed_states) // 20)  # Draw ~20 robot rectangles
            
            for i in range(0, len(executed_states), step_interval):
                state = executed_states[i]
                x, y, theta = state[0], state[1], state[2]
                
                # Create rotated rectangle centered at robot position
                rect = patches.Rectangle(
                    (-robot_length/2, -robot_width/2), robot_length, robot_width,
                    linewidth=1.5, edgecolor='darkblue', facecolor='cyan', alpha=0.3
                )
                
                # Create transform: rotate and translate
                t = Affine2D().rotate_around(0, 0, theta) + Affine2D().translate(x, y)
                t += ax1.transData
                rect.set_transform(t)
                ax1.add_patch(rect)
        
        # Start and goal
        ax1.scatter(*self.start[:2], s=200, c='green', marker='o', edgecolors='darkgreen', linewidth=2, label='Start', zorder=10)
        ax1.scatter(*self.goal[:2], s=200, c='red', marker='s', edgecolors='darkred', linewidth=2, label='Goal', zorder=10)
        
        ax1.set_xlabel('X (meters)', fontsize=12)
        ax1.set_ylabel('Y (meters)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # --- Error Plots ---
        # Calculate path errors (x, y, theta relative to closest path point)
        error_x, error_y, error_theta = self.calculate_path_errors(executed_states, path)
        cross_track_errors = self.calculate_cross_track_error(executed_states, path)
        time_steps = np.arange(len(executed_states))
        
        if self.DEBUG:
            print(f"\n[DEBUG] Path errors calculated: {len(error_x)} points")
        
        # X error plot (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(time_steps, error_x, 'r-', linewidth=2, label='X Error')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.fill_between(time_steps, 0, error_x, alpha=0.3, color='red')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Error (meters)', fontsize=12)
        ax2.set_title('X Position Error vs Path', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Y error plot (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_steps, error_y, 'b-', linewidth=2, label='Y Error')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.fill_between(time_steps, 0, error_y, alpha=0.3, color='blue')
        ax3.set_xlabel('Time Step', fontsize=12)
        ax3.set_ylabel('Error (meters)', fontsize=12)
        ax3.set_title('Y Position Error vs Path', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Theta error plot (bottom middle)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_steps, error_theta, 'g-', linewidth=2, label='Theta Error')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.fill_between(time_steps, 0, error_theta, alpha=0.3, color='green')
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_ylabel('Error (radians)', fontsize=12)
        ax4.set_title('Theta (Angle) Error vs Path', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Cross-track error plot (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(time_steps, cross_track_errors, 'm-', linewidth=2, label='Cross-Track Error')
        ax5.fill_between(time_steps, 0, cross_track_errors, alpha=0.3, color='magenta')
        ax5.set_xlabel('Time Step', fontsize=12)
        ax5.set_ylabel('Error (meters)', fontsize=12)
        ax5.set_title('Cross-Track Error (Distance to Path)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
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
        
    def run(self):
        """Run complete controller test."""
        print("=" * 60)
        print("CONTROLLER TEST")
        print("=" * 60)
        
        # Plan path
        path = self.plan_path()
        if path is None:
            return
        
        # Generate trajectory
        ref_traj: np.ndarray = self.generate_trajectory(path)
        
        # Simulate controller
        executed_states, executed_controls, reference_states = self.simulate_controller(path, ref_traj)
        
        # Visualize
        print("\nVisualizing results...")
        self.visualize(path, executed_states, ref_traj, reference_states)
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    try:
        tester = ControllerTester()
        tester.run()
    except Exception as e:
        print(f"\n✗ Error during controller testing: {e}")
        import traceback
        traceback.print_exc()

