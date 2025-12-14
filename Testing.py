"""
Test file for Path Planners and MPPI Controller
Tests the Planning algorithms and Controller with the robot's map and obstacle detector.
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

import json
import math
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PathPlanning.AStarPlanner import AStarPlanner
from ObstacleDetection.ObstacleDetector import StaticObstacleChecker
from PathPlanning.Planners import Planner, PlannerTypes, smooth_path
from PathPlanning.RRT_StarPlanner import RRTStarPlanner
from PathPlanning.HybridAStarPlanner import HybridAStarPlanner
from PathPlanning.DStarPlanner import DStarPlanner
from PathPlanning.TrajectoryGenerator import TrajectoryGenerator
from PathController.Robot_Sim import Robot_Sim
from PathController.MPPI.MPPI_Controller import MPPIController
from PathController.Controller import Controller, ControllerTypes
from PathController.SMC_Controller import SMCController
from PathController.LQR_Controller import LQRController
from PathController.MRAC_Controller import MRACController
from ActuatorController.ActuatorController import ActuatorController
from PathController.PathReference import ProjectedPathFollower
from Scene.JsonManager import load_json
from Scene.Map import Map
from Scene.Robot import Robot
from Types import State2D, ConvexShape
import numpy as np


def _get_rotated_robot_corners(center_x: float, center_y: float, length: float, width: float, theta: float) -> List[Tuple[float, float]]:
    """
    Calculate the corners of a rotated rectangle representing the robot.
    
    Args:
        center_x, center_y: Center position of robot
        length: Robot length (along x-axis when theta=0)
        width: Robot width (along y-axis when theta=0)
        theta: Rotation angle in radians
        
    Returns:
        List of (x, y) corners in world frame
    """
    half_length = length / 2.0
    half_width = width / 2.0
    
    # Corners in local frame (centered at origin)
    corners_local = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width)
    ]
    
    # Rotate and translate to world frame
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    corners_world = []
    for lx, ly in corners_local:
        # Rotate
        wx = cos_theta * lx - sin_theta * ly
        wy = sin_theta * lx + cos_theta * ly
        # Translate
        wx += center_x
        wy += center_y
        corners_world.append((wx, wy))
    
    return corners_world


def visualize_path_on_map(map_obj: Map, robot: Robot, obstacles, path: List[State2D], start: State2D, goal: State2D, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                          title: str = "Path Planning Result"):
    """Visualize the planned path with obstacles on the map."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set map boundaries
    x_min, x_max = world_bounds[0]
    y_min, y_max = world_bounds[1]
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_aspect('equal')
    
    # Draw map boundary
    boundary = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='black', facecolor='none', label='Map Boundary'
    )
    ax.add_patch(boundary)
    
    # Draw obstacles
    for i, obstacle in enumerate(obstacles):
        if obstacle.shape == ConvexShape.Circle:
            circle = patches.Circle(
                (obstacle.center[0], obstacle.center[1]),
                obstacle.radius,
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
                label='Obstacles' if i == 0 else ''
            )
            ax.add_patch(circle)
        elif obstacle.shape == ConvexShape.Polygon:
            if hasattr(obstacle, 'points') and obstacle.points:
                polygon = patches.Polygon(
                    obstacle.points,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
                    label='Obstacles' if i == 0 else ''
                )
                ax.add_patch(polygon)
    
    # Draw path if it exists
    if path:
        path_x = [state[0] for state in path]
        path_y = [state[1] for state in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path', zorder=5)
        
        # Draw waypoints
        ax.plot(path_x, path_y, 'bo', markersize=4, zorder=5)
        
        # Draw orientation arrows at selected waypoints
        step = max(1, len(path) // 10)  # Show ~10 arrows
        for i in range(0, len(path), step):
            state = path[i]
            x, y, theta = state
            # Arrow length proportional to robot size
            arrow_len = 0.5
            dx = arrow_len * math.cos(theta)
            dy = arrow_len * math.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2, 
                    fc='blue', ec='blue', alpha=0.6, zorder=5)
    
    # Draw start and goal
    ax.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=10)
    ax.arrow(start[0], start[1], 
            0.5 * math.cos(start[2]), 0.5 * math.sin(start[2]),
            head_width=0.3, head_length=0.2, fc='green', ec='green', zorder=10)
    
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=10)
    ax.arrow(goal[0], goal[1], 
            0.5 * math.cos(goal[2]), 0.5 * math.sin(goal[2]),
            head_width=0.3, head_length=0.2, fc='orange', ec='orange', zorder=10)
    
    # Draw robot footprint at start position
    start_corners = _get_rotated_robot_corners(start[0], start[1], robot.length, robot.width, start[2])
    start_robot_polygon = patches.Polygon(
        start_corners,
        linewidth=2, edgecolor='green', facecolor='green', 
        alpha=0.2, zorder=4, label='Robot at Start'
    )
    ax.add_patch(start_robot_polygon)
    
    # Draw robot footprint at goal position
    goal_corners = _get_rotated_robot_corners(goal[0], goal[1], robot.length, robot.width, goal[2])
    goal_robot_polygon = patches.Polygon(
        goal_corners,
        linewidth=2, edgecolor='red', facecolor='red', 
        alpha=0.2, zorder=4, label='Robot at Goal'
    )
    ax.add_patch(goal_robot_polygon)
    
    # Labels and legend
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig, ax


def test_planner():
    """Test the planner with map and obstacle detector."""
    
    # Load configuration files
    print("Loading configuration...")
    config_path = "Scene/Configuration.json"
    param_path = "Scene/Parameters.json"
    config = load_json(config_path)
    params = load_json(param_path)
    
    # Extract robot and map config
    robot_config = config.get("Robot", {})
    map_config = config.get("Map", {})
    
    print(f"  Robot dimensions: {robot_config['Dimensions']}")
    print(f"  Map dimensions: {map_config['Dimensions']}")
    
    # Create Robot and Map objects
    robot = Robot(robot_config)
    map_obj = Map(map_config)
    
    # Define world bounds from map dimensions
    world_bounds = ((0, map_obj.length),(0, map_obj.width))
    
    print(f"\nWorld bounds: x=[0, {map_obj.length}], y=[0, {map_obj.width}]")
    print(f"Number of obstacles: {len(map_obj.obstacles)}")
    
    # Debug: Print obstacle details
    for i, obs in enumerate(map_obj.obstacles):
        if obs.shape == ConvexShape.Circle:
            print(f"  Obstacle {i}: Circle at {obs.center}, radius={obs.radius}")
        elif obs.shape == ConvexShape.Polygon:
            print(f"  Obstacle {i}: Polygon with {len(obs.points)} vertices")
    
    # Create obstacle checker
    obstacle_checker = StaticObstacleChecker(robot=robot, map_limits=world_bounds, obstacles=map_obj.obstacles, use_parallelization=False)
    
    # Create Path planner
    planner_params = params["Path Planning"]
    planner_type_str = planner_params["Planner"]["type"]
    try:
        planner_type = PlannerTypes(planner_type_str)
    except (ValueError, KeyError):
        print(f"Warning: Unknown planner type '{planner_type_str}', using AStarPlanner")
        planner_type = PlannerTypes.AStarPlanner
    planner_params = params["Path Planning"][planner_type.value]
    if planner_type == PlannerTypes.AStarPlanner:
        grid_resolution: float = planner_params["grid_resolution"]
        angle_resolution: float = planner_params["angle_resolution"]
        planner: Planner = AStarPlanner(obstacle_checker=obstacle_checker, world_bounds=world_bounds, grid_resolution=grid_resolution, angle_resolution=angle_resolution)
    elif planner_type == PlannerTypes.HybridAStarPlanner:  
        grid_resolution: float = planner_params["grid_resolution"]
        angle_bins: int = planner_params["angle_bins"]      
        planner: Planner = HybridAStarPlanner(obstacle_checker=obstacle_checker, world_bounds=world_bounds, grid_resolution=grid_resolution, angle_bins=angle_bins)
    elif planner_type == PlannerTypes.RRTStarPlanner:
        rewire_radius_factor = planner_params["rewire_radius_factor"]
        rrt_timeout = planner_params["timeout"]
        planner: Planner = RRTStarPlanner(obstacle_checker=obstacle_checker, world_bounds=world_bounds, timeout=rrt_timeout, rewire_radius_factor=rewire_radius_factor)
    else: #if planner_type == PlannerTypes.DStarPlanner:
        grid_resolution: float = planner_params["grid_resolution"]
        angle_resolution: float = planner_params["angle_resolution"]
        planner: Planner = DStarPlanner(obstacle_checker=obstacle_checker, world_bounds=world_bounds, grid_resolution=grid_resolution, angle_resolution=angle_resolution)
    
    # Test cases: (start, goal, description)
    test_cases = [
        (
            (5.0, 5.0, 0.0),
            (map_obj.length - 20.0, map_obj.width - 20.0, 0.0),
            "Corner to opposite corner"
        ),
        (
            (map_obj.length / 2, map_obj.width / 2, 0.0),
            (map_obj.length / 2 + 2.0, map_obj.width / 2 + 2.0, math.pi / 4),
            "Center to nearby position"
        ),
        (
            (5.0, map_obj.width / 2, math.pi / 2),
            (map_obj.length - 20.0,  map_obj.width - 20.0, math.pi / 2),
            "Left to right side"
        ),
    ]
    
    # Run tests
    for i, (start, goal, description) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {description}")
        print(f"{'='*60}")
        print(f"Start: x={start[0]:.2f}, y={start[1]:.2f}, θ={start[2]:.2f} rad ({math.degrees(start[2]):.1f}°)")
        print(f"Goal:  x={goal[0]:.2f}, y={goal[1]:.2f}, θ={goal[2]:.2f} rad ({math.degrees(goal[2]):.1f}°)")
        
        # Debug: Check if start and goal are in collision
        start_collision = obstacle_checker.is_collision(start)
        goal_collision = obstacle_checker.is_collision(goal)
        print(f"\nDebug Info:")
        print(f"  Start in collision: {start_collision}")
        print(f"  Goal in collision: {goal_collision}")
        
        if start_collision or goal_collision:
            print(f"  ⚠️  Start or goal is in collision! Skipping this test.")
            continue
               
        # Plan path
        print(f"\n  Planning path...")
        path = planner.plan(start, goal)
        path = smooth_path(path, obstacle_checker)  # type: ignore # Smooth the path if found
        
        if path:
            print(f"\n✓ Path found with {len(path)} waypoints")
            print(f"\nPath waypoints:")
            for j, state in enumerate(path):
                print(f"  {j:3d}: x={state[0]:7.2f}, y={state[1]:7.2f}, θ={math.degrees(state[2]):7.1f}°")
            
            # Calculate total path length
            total_length = 0.0
            for j in range(1, len(path)):
                dx = path[j][0] - path[j-1][0]
                dy = path[j][1] - path[j-1][1]
                total_length += math.sqrt(dx**2 + dy**2)
            print(f"\nTotal path length: {total_length:.2f} units")
            
            # Visualize the path
            fig, ax = visualize_path_on_map(
                map_obj, robot, map_obj.obstacles, path, start, goal, world_bounds,
                title=f"A* Path: {description}"
            )
            plt.show()
        else:
            print(f"\n✗ No path found!")
            print(f"  Debug: Max iterations may have been reached")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


def test_collision_detection():
    """Test collision detection with specific points."""
    
    print("\n" + "="*60)
    print("Collision Detection Test")
    print("="*60)
    
    config = load_json("Scene/Configuration.json")
    robot = Robot(config.get("Robot", {}))
    map_obj = Map(config.get("Map", {}))
    
    world_bounds = (
        (0, map_obj.length),
        (0, map_obj.width)
    )
    
    obstacle_checker = StaticObstacleChecker(
        robot=robot,
        map_limits=world_bounds,
        obstacles=map_obj.obstacles,
        use_parallelization=False
    )
    
    # Test points
    test_points = [
        ((5.0, 5.0, 0.0), "Free space (corner)"),
        ((map_obj.length / 2, map_obj.width / 2, 0.0), "Center"),
        ((0.1, 0.1, 0.0), "Near boundary"),
    ]
    
    print(f"\nTesting {len(test_points)} points for collisions:\n")
    for point, description in test_points:
        is_collision = obstacle_checker.is_collision(point)
        status = "COLLISION" if is_collision else "CLEAR"
        print(f"  {description:30s} -> {status}")


class ControllerTester:
    """Test MPPI controller with planned trajectories."""
    DEBUG = True  # Set to False to remove all debug prints
    
    def __init__(self, config_path: str = "Scene/Configuration.json", params_path: str = "Scene/Parameters.json"):
        """Initialize controller tester."""
        self.config = load_json(config_path)
        self.params = load_json(params_path)
        
        # Create robot and map
        self.robot = Robot(self.config.get("Robot", {}))
        self.map_obj = Map(self.config.get("Map", {}))
        self.world_bounds = ((0, self.map_obj.length), (0, self.map_obj.width))
        
        # Create obstacle checker
        self.obstacle_checker = StaticObstacleChecker(
            robot=self.robot,
            obstacles=self.map_obj.obstacles,
            map_limits=self.world_bounds,
            use_parallelization=False
        )
        
        # Create actuator controller
        self.actuator_controller = ActuatorController(self.robot)
        
        # Use same test points as path planner tests
        self.start = (5.0, 5.0, 0.0)
        self.goal = (self.map_obj.length - 20.0, self.map_obj.width - 20.0, 0.0)
    
    def get_planner(self):
        """Get planner based on parameters."""
        planner_config = self.params.get("Path Planning", {})
        planner_type = planner_config.get("Planner", {}).get("type", "RRTStarPlanner")
        
        print(f"Creating planner: {planner_type}")
        
        if planner_type == "AStarPlanner":
            params = planner_config.get("AStarPlanner", {})
            return AStarPlanner(
                grid_resolution=params.get("grid_resolution", 1.0),
                angle_resolution=params.get("angle_resolution", np.pi/4),
                obstacle_checker=self.obstacle_checker,
                world_bounds=self.world_bounds
            )
        
        elif planner_type == "RRTStarPlanner":
            params = planner_config.get("RRTStarPlanner", {})
            return RRTStarPlanner(
                obstacle_checker=self.obstacle_checker,
                world_bounds=self.world_bounds,
                step_size=params.get("step_size", 1.0),
                goal_sample_rate=params.get("goal_sample_rate", 0.1),
                rewire_radius_factor=params.get("rewire_radius_factor", 10.0),
                timeout=params.get("timeout", 10.0)
            )
        
        elif planner_type == "HybridAStarPlanner":
            params = planner_config.get("HybridAStarPlanner", {})
            return HybridAStarPlanner(
                grid_resolution=params.get("grid_resolution", 0.5),
                angle_bins=params.get("angle_bins", 16),
                obstacle_checker=self.obstacle_checker,
                world_bounds=self.world_bounds
            )
        
        elif planner_type == "DStarPlanner":
            params = planner_config.get("DStarPlanner", {})
            return DStarPlanner(
                grid_resolution=params.get("grid_resolution", 2.0),
                angle_resolution=params.get("angle_resolution", np.pi/4),
                obstacle_checker=self.obstacle_checker,
                world_bounds=self.world_bounds
            )
        
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
    
    def plan_path(self):
        """Plan path using configured planner."""
        planner = self.get_planner()
        print(f"\nPlanning path from {self.start} to {self.goal}...")
        path = planner.plan(self.start, self.goal)
        
        if path is None:
            print("❌ No path found!")
            return None
        
        print(f"[OK] Path found with {len(path)} waypoints")
        return path
    
    def generate_trajectory(self, path: List[State2D]) -> np.ndarray:
        """Generate reference trajectory from path."""
        control_config = self.params.get("Control", {})
        mppi_config = control_config.get("MPPI", {})
        lqr_config = control_config.get("LQR", {})
        
        dt = mppi_config.get("dt", 0.1)  # Use same dt as simulation
        # Calculate horizon needed to traverse the entire path
        # Path length estimate: sum of distances between waypoints
        path_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += (dx**2 + dy**2)**0.5
        
        # Use actual controller velocity instead of max velocity
        v_desired = lqr_config.get("v_desired", 2.0)  # m/s
        # Account for obstacle-aware velocity scaling - robot may move at min_velocity_scale (20%) near obstacles
        # Use a conservative average velocity estimate (30% of desired) to ensure enough simulation time
        min_velocity_scale = 0.2  # From PathReference default
        avg_velocity_estimate = v_desired * 0.3  # Conservative: assume 30% average speed due to obstacles
        time_to_goal = path_length / avg_velocity_estimate  # seconds
        horizon = int(time_to_goal / dt) + 100  # Add 100 extra steps as buffer
        horizon = max(horizon, 500)  # At least 500 steps
        
        print(f"  Path length: {path_length:.2f}m, time to goal: {time_to_goal:.2f}s, horizon: {horizon}")
        
        traj_gen = TrajectoryGenerator(dt=dt, horizon=horizon, max_velocity=v_desired)
        
        # Initial state: start position with zero velocity
        current_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print(f"\nGenerating reference trajectory (horizon={horizon}, dt={dt}, total_time={horizon*dt}s)...")
        ref_traj = traj_gen.get_reference_trajectory(current_state, path)
        
        print(f"[OK] Reference trajectory shape: {ref_traj.shape}")
        print(f"  Ref positions (first 5): {ref_traj[:2, :5]}")
        print(f"  Ref velocities (first 5): {ref_traj[3:6, :5]}")
        print(f"  Ref positions (last 5): {ref_traj[:2, -5:]}")
        print(f"  Ref velocities (last 5): {ref_traj[3:6, -5:]}")
        
        # Debug: Check path endpoints vs original goal
        print(f"\n[DEBUG] Path endpoint check:")
        print(f"  Original goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        print(f"  Path last point: ({path[-1][0]:.2f}, {path[-1][1]:.2f})")
        print(f"  Trajectory final position: ({ref_traj[0, -1]:.2f}, {ref_traj[1, -1]:.2f})")
        
        return ref_traj
    
    def simulate_controller(self, path, ref_traj: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Simulate MPPI controller following reference trajectory with rolling horizon."""
        # Debug: Check path endpoint
        print(f"\n[DEBUG] PathFollower path check:")
        print(f"  Original goal: ({self.goal[0]:.4f}, {self.goal[1]:.4f})")
        print(f"  Path last point: ({path[-1][0]:.4f}, {path[-1][1]:.4f})")
        
        # Create Controller
        controller_params = self.params["Control"]
        controller_type_str = controller_params["Controller"]["type"]
        try:
            controller_type = ControllerTypes(controller_type_str)
        except (ValueError, KeyError):
            print(f"Warning: Unknown controller type '{controller_type_str}', using LQRController")
            controller_type = ControllerTypes.LQR
        controller_params = controller_params[controller_type.value]
        # Create robot simulator
        dt: float = controller_params["dt"]
        robot_sim = Robot_Sim(self.actuator_controller, self.robot, dt=dt)
        path_follower = ProjectedPathFollower(path_points=path, obstacle_checker=self.obstacle_checker)
        if controller_type == ControllerTypes.LQR:
            lookahead: float = controller_params["lookahead"]
            v_desired: float = controller_params["v_desired"]
            dt: float = controller_params["dt"]
            Q = controller_params["Q"]
            R = controller_params["R"]
            controller: Controller = LQRController(robot_controller= self.actuator_controller, path_follower=path_follower, lookahead=lookahead, v_desired=v_desired, dt=dt, Q=Q, R=R)
        elif controller_type == ControllerTypes.MRAC:
            lookahead: float = controller_params["lookahead"]
            v_desired: float = controller_params["v_desired"]
            dt: float = controller_params["dt"]
            gamma: float = controller_params["gamma"]
            kp: float = controller_params["kp"]
            kv: float = controller_params["kv"]
            alpha_min: float = controller_params["alpha_min"]
            alpha_max: float = controller_params["alpha_max"]
            controller: Controller = MRACController(robot_controller= self.actuator_controller, path_follower=path_follower, lookahead=lookahead, v_desired=v_desired, dt=dt, gamma=gamma, kp=kp, kv=kv, alpha_min=alpha_min, alpha_max=alpha_max)
        else: #MPPI
            controller: Controller = MPPIController(desired_traj=path, robot_sim=robot_sim, collision_check_method=self.obstacle_checker.is_collision, N_Horizon=controller_params["N_Horizon"], lambda_=controller_params["Lambda"], myu=controller_params["myu"], K=controller_params["K"])
        
        
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
            print(f"[DEBUG] Reference trajectory shape: {ref_traj.shape}")
            print(f"[DEBUG] Ref traj first 3 elements (x,y,theta): {ref_traj[:3, 0]}")
        
        print(f"\nSimulating controller for {ref_traj.shape[1]} steps...")
        for step in range(ref_traj.shape[1] - 1):
            # Get reference state from path follower for debugging
            ref_state = controller.path_follower.get_reference_state(
                np.array([current_state[0], current_state[1], current_state[2]]), 
                controller._lookahead, 
                controller._v_desired
            )
            reference_states.append(ref_state.copy())
            
            # Compute control from controller
            control_input = controller.get_command(current_state)
            executed_controls.append(control_input.copy())
            
            # Propagate state using robot simulator
            current_state = robot_sim.propagate(current_state, control_input)
            executed_states.append(current_state.copy())
            
            if (step + 1) % 100 == 0:
                speed = np.sqrt(current_state[3]**2 + current_state[4]**2)
                print(f"  Step {step + 1}/{ref_traj.shape[1] - 1}: Position ({current_state[0]:.2f}, {current_state[1]:.2f}), Speed {speed:.3f} m/s")
        
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
            px = closest_path_point[0]
            py = closest_path_point[1]
            p_theta = closest_path_point[2] if len(closest_path_point) > 2 else 0
            
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
    
    def visualize(self, path: List[State2D], executed_states: List[np.ndarray], ref_traj: np.ndarray, reference_states: np.ndarray = None):
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
                    (obstacle.center[0], obstacle.center[1]), obstacle.radius,
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
        ref_x = ref_traj[0, :]
        ref_y = ref_traj[1, :]
        ax1.plot(ref_x, ref_y, 'g--', linewidth=1, label='Reference Trajectory', alpha=0.7)
        
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
            robot_length = self.robot.length
            robot_width = self.robot.width
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
                t = patches.transforms.Affine2D().rotate_around(0, 0, theta) + patches.transforms.Affine2D().translate(x, y)
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
        ref_traj = self.generate_trajectory(path)
        
        # Simulate controller
        executed_states, executed_controls, reference_states = self.simulate_controller(path, ref_traj)
        
        # Visualize
        print("\nVisualizing results...")
        self.visualize(path, executed_states, ref_traj, reference_states)
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)


def test_controller():
    """Test controller with path planning and trajectory generation."""
    tester = ControllerTester()
    tester.run()


def show_menu():
    """Display menu and get user choice."""
    print("\n" + "=" * 60)
    print("AUTONOMOUS VEHICLE TESTING SUITE")
    print("=" * 60)
    print("\nSelect test to run:")
    print("  1 - Path Planner Tests")
    print("  2 - Collision Detection Tests")
    print("  3 - MPPI Controller Test")
    print("  4 - Run All Tests")
    print("  0 - Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-4): ").strip()
            if choice in ['0', '1', '2', '3', '4']:
                return choice
            else:
                print("Invalid choice. Please enter 0-4.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return '0'


if __name__ == "__main__":
    try:
        test_controller()
    except Exception as e:
        print(f"\n✗ Error during controller testing: {e}")
        import traceback
        traceback.print_exc()

