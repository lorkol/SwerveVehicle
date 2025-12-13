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
from ActuatorController.ActuatorController import ActuatorController
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
        
        dt = mppi_config.get("dt", 0.1)  # Use same dt as simulation
        # Calculate horizon needed to traverse the entire path
        # Path length estimate: sum of distances between waypoints
        path_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += (dx**2 + dy**2)**0.5
        
        max_velocity = 2.0  # m/s
        time_to_goal = path_length / max_velocity  # seconds
        horizon = int(time_to_goal / dt) + 50  # Add 50 extra steps as buffer
        horizon = max(horizon, 500)  # At least 500 steps
        
        print(f"  Path length: {path_length:.2f}m, time to goal: {time_to_goal:.2f}s, horizon: {horizon}")
        
        traj_gen = TrajectoryGenerator(dt=dt, horizon=horizon, max_velocity=max_velocity)
        
        # Initial state: start position with zero velocity
        current_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print(f"\nGenerating reference trajectory (horizon={horizon}, dt={dt}, total_time={horizon*dt}s)...")
        ref_traj = traj_gen.get_reference_trajectory(current_state, path)
        
        print(f"[OK] Reference trajectory shape: {ref_traj.shape}")
        print(f"  Ref positions (first 5): {ref_traj[:2, :5]}")
        print(f"  Ref velocities (first 5): {ref_traj[3:6, :5]}")
        print(f"  Ref positions (last 5): {ref_traj[:2, -5:]}")
        print(f"  Ref velocities (last 5): {ref_traj[3:6, -5:]}")
        return ref_traj
    
    def simulate_controller(self, ref_traj: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Simulate MPPI controller following reference trajectory with rolling horizon."""
        control_config = self.params.get("Control", {})
        mppi_config = control_config.get("MPPI", {})
        dt = mppi_config.get("dt", 0.1)
        mppi_horizon = 10  # MPPI horizon for lookahead
        
        # Create robot simulator
        robot_sim = Robot_Sim(self.actuator_controller, self.robot, dt=dt)
        
        # Initial state
        initial_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot_sim.set_state(initial_state)
        
        # Simulate
        print(f"Simulating controller (MPPI horizon={mppi_horizon})...")
        executed_states = [initial_state.copy()]
        executed_controls = []
        
        max_steps = ref_traj.shape[1] * 20  # Allow twice the reference trajectory length
        goal_threshold = 1.0  # Distance to goal
        ref_idx = 0  # Current index in reference trajectory
        
        for step in range(max_steps):
            current_state = executed_states[-1]
            
            # Check if reached goal
            dist_to_goal = np.linalg.norm(current_state[:2] - np.array(self.goal[:2]))
            if dist_to_goal < goal_threshold:
                print(f"[OK] Reached goal at step {step}")
                break
            
            # Find closest reference point to current position
            # Compute distances from current position to all reference positions
            ref_positions = ref_traj[0:2, :]  # Shape: (2, N_traj)
            current_pos = current_state[0:2]  # Shape: (2,)
            distances = np.linalg.norm(ref_positions - current_pos[:, np.newaxis], axis=0)  # Shape: (N_traj,)
            closest_idx = np.argmin(distances)
            
            # Use closest index as the start of the rolling window (look ahead some steps)
            # Don't allow going backward in the trajectory
            ref_idx = max(ref_idx, closest_idx)
            
            # Debug: show current position vs reference position
            if (step + 1) % 20 == 0:
                ref_pos_at_idx = ref_traj[0:2, ref_idx]
                print(f"  Step {step}: robot at ({current_pos[0]:.2f}, {current_pos[1]:.2f}), closest ref idx={closest_idx}, using idx={ref_idx}, ref pos=({ref_pos_at_idx[0]:.2f}, {ref_pos_at_idx[1]:.2f})")
            
            # Extract rolling window of reference trajectory (5 steps ahead)
            ref_end_idx = min(ref_idx + mppi_horizon, ref_traj.shape[1])
            ref_window_size = ref_end_idx - ref_idx
            
            # Pad with goal state if we've used most of the trajectory
            rolling_ref = np.zeros((ref_traj.shape[0], mppi_horizon))
            rolling_ref[:, :ref_window_size] = ref_traj[:, ref_idx:ref_end_idx]
            
            # Fill remaining steps with goal state (extrapolate)
            if ref_window_size < mppi_horizon:
                rolling_ref[:, ref_window_size:] = ref_traj[:, -1:].repeat(mppi_horizon - ref_window_size, axis=1)
            
            # Debug output for first few steps
            if step < 3:
                print(f"  Step {step}: rolling_ref[0:2,0:3] = {rolling_ref[0:2, :3]}")
            
            # Create MPPI controller for this iteration with rolling window
            mppi_controller = MPPIController(
                desired_traj=rolling_ref,
                robot_sim=robot_sim,
                collision_check_method=lambda state: False,  # DISABLE collision checking for testing
                N_Horizon=mppi_horizon,
                lambda_=1.0,
                myu=20.0,
                K=50
            )
            
            # Get control command from MPPI
            try:
                control = mppi_controller.get_command(current_state)
                executed_controls.append(control)
                if (step + 1) % 100 == 0 and np.any(control != 0):
                    print(f"  Step {step}: control={control[:4]}")
                elif (step + 1) % 100 == 0:
                    print(f"  Step {step}: WARNING - zero control!")
                if (step + 1) % 20 == 0:
                    print(f"  Step {step}: current_pos=({current_state[0]:.2f}, {current_state[1]:.2f})")
                    print(f"    ref_idx={ref_idx}, rolling_ref first 3 steps x: {rolling_ref[0, :3]}")
                    print(f"    control torques={control[:4]}, steering_rates={control[4:]}")
            except Exception as e:
                print(f"[ERROR] Error getting control at step {step}: {e}")
                break
            
            # Propagate state
            try:
                next_state = robot_sim.propagate(current_state, control)
                
                # Check collision - DISABLED FOR TESTING
                # if self.obstacle_checker.is_collision(tuple(next_state[:3])):
                #     print(f"[ERROR] Collision detected at step {step}")
                #     break
                
                executed_states.append(next_state)
                # ref_idx is now computed dynamically from closest point, no need to manually increment
                
                if (step + 1) % 10 == 0:
                    print(f"  Step {step + 1}: pos=({next_state[0]:.2f}, {next_state[1]:.2f}), dist_to_goal={dist_to_goal:.2f}m, ref_idx={ref_idx}")
            
            except Exception as e:
                print(f"[ERROR] Error propagating state at step {step}: {e}")
                break
        
        print(f"[OK] Simulation completed: {len(executed_states)} states")
        
        # Calculate and display velocity statistics
        print(f"\n=== VELOCITY COMPARISON ===")
        if len(executed_states) > 1:
            # Calculate actual velocities
            actual_vx = []
            actual_vy = []
            actual_speed = []
            for i in range(1, len(executed_states)):
                dx = executed_states[i][0] - executed_states[i-1][0]
                dy = executed_states[i][1] - executed_states[i-1][1]
                vx = dx / dt
                vy = dy / dt
                speed = np.sqrt(vx**2 + vy**2)
                actual_vx.append(vx)
                actual_vy.append(vy)
                actual_speed.append(speed)
            
            # Reference velocities (from trajectory)
            ref_vx = ref_traj[3, :len(executed_states)-1]
            ref_vy = ref_traj[4, :len(executed_states)-1]
            ref_speed = np.sqrt(ref_vx**2 + ref_vy**2)
            
            # Statistics
            print(f"Actual avg speed: {np.mean(actual_speed):.3f} m/s (max: {np.max(actual_speed):.3f})")
            print(f"Reference avg speed: {np.mean(ref_speed):.3f} m/s (max: {np.max(ref_speed):.3f})")
            print(f"Speed ratio (actual/reference): {np.mean(actual_speed) / np.mean(ref_speed):.1%}")
            print(f"\nFirst 10 step comparison (actual vs reference):")
            print(f"  Step | Actual Speed | Ref Speed | Actual vx | Ref vx")
            for i in range(min(10, len(actual_speed))):
                print(f"  {i:3d} | {actual_speed[i]:12.3f} | {ref_speed[i]:9.3f} | {actual_vx[i]:9.3f} | {ref_vx[i]:6.3f}")
        
        return executed_states, executed_controls
    
    def visualize(self, path: List[State2D], executed_states: List[np.ndarray], ref_traj: np.ndarray):
        """Visualize planning and control results."""
        print(f"\nVisualizing: {len(executed_states)} executed states")
        if executed_states:
            print(f"  First state: ({executed_states[0][0]:.2f}, {executed_states[0][1]:.2f})")
            print(f"  Last state: ({executed_states[-1][0]:.2f}, {executed_states[-1][1]:.2f})")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        (x_min, x_max), (y_min, y_max) = self.world_bounds
        
        # --- Plot 1: Path Planning ---
        ax1.set_xlim(x_min - 1, x_max + 1)
        ax1.set_ylim(y_min - 1, y_max + 1)
        ax1.set_aspect('equal')
        ax1.set_title('Path Planning Result', fontsize=14, fontweight='bold')
        
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
        
        # Planned path
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax1.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
            ax1.scatter(path_x, path_y, s=20, c='blue', zorder=5)
        
        # Start and goal
        ax1.scatter(*self.start[:2], s=200, c='green', marker='o', edgecolors='darkgreen', linewidth=2, label='Start', zorder=10)
        ax1.scatter(*self.goal[:2], s=200, c='red', marker='s', edgecolors='darkred', linewidth=2, label='Goal', zorder=10)
        
        ax1.set_xlabel('X (meters)', fontsize=12)
        ax1.set_ylabel('Y (meters)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Controller Execution ---
        ax2.set_xlim(x_min - 1, x_max + 1)
        ax2.set_ylim(y_min - 1, y_max + 1)
        ax2.set_aspect('equal')
        ax2.set_title('Controller Execution Trajectory', fontsize=14, fontweight='bold')
        
        # Map boundary
        boundary2 = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.1
        )
        ax2.add_patch(boundary2)
        
        # Obstacles
        for i, obstacle in enumerate(self.map_obj.obstacles):
            if obstacle.shape == ConvexShape.Circle:
                circle = patches.Circle(
                    (obstacle.center[0], obstacle.center[1]), obstacle.radius,
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.4,
                    label='Obstacles' if i == 0 else ''
                )
                ax2.add_patch(circle)
            elif obstacle.shape == ConvexShape.Polygon:
                polygon = patches.Polygon(
                    obstacle.points, linewidth=2, edgecolor='red', facecolor='red', alpha=0.4,
                    label='Obstacles' if i == 0 else ''
                )
                ax2.add_patch(polygon)
        
        # Reference trajectory (first 3 states for visualization)
        ref_x = ref_traj[0, :]
        ref_y = ref_traj[1, :]
        ax2.plot(ref_x, ref_y, 'g--', linewidth=1, label='Reference Trajectory', alpha=0.7)
        
        # Executed trajectory
        if executed_states:
            exec_x = [state[0] for state in executed_states]
            exec_y = [state[1] for state in executed_states]
            ax2.plot(exec_x, exec_y, 'b-', linewidth=2, label='Executed Trajectory')
            ax2.scatter(exec_x, exec_y, s=10, c='blue', alpha=0.5, zorder=5)
        
        # Start and goal
        ax2.scatter(*self.start[:2], s=200, c='green', marker='o', edgecolors='darkgreen', linewidth=2, label='Start', zorder=10)
        ax2.scatter(*self.goal[:2], s=200, c='red', marker='s', edgecolors='darkred', linewidth=2, label='Goal', zorder=10)
        
        ax2.set_xlabel('X (meters)', fontsize=12)
        ax2.set_ylabel('Y (meters)', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save figure instead of blocking on show()
        plt.savefig('controller_test_result.png', dpi=150, bbox_inches='tight')
        print("Figure saved to controller_test_result.png")
        plt.show()
    
    def run(self):
        """Run complete controller test."""
        print("=" * 60)
        print("MPPI CONTROLLER TEST")
        print("=" * 60)
        
        # Plan path
        path = self.plan_path()
        if path is None:
            return
        
        # Generate trajectory
        ref_traj = self.generate_trajectory(path)
        
        # Simulate controller
        executed_states, executed_controls = self.simulate_controller(ref_traj)
        
        # Visualize
        print("\nVisualizing results...")
        self.visualize(path, executed_states, ref_traj)
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)


def test_controller():
    """Test MPPI controller with path planning and trajectory generation."""
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
    while True:
        choice = show_menu()
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            try:
                test_planner()
            except Exception as e:
                print(f"\n✗ Error during planner testing: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '2':
            try:
                test_collision_detection()
            except Exception as e:
                print(f"\n✗ Error during collision detection testing: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '3':
            try:
                test_controller()
            except Exception as e:
                print(f"\n✗ Error during controller testing: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '4':
            try:
                test_planner()
                test_collision_detection()
                test_controller()
            except Exception as e:
                print(f"\n✗ Error during testing: {e}")
                import traceback
                traceback.print_exc()

