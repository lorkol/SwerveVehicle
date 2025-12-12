"""
Test file for A* Path Planner
Tests the A* algorithm with the robot's map and obstacle detector.
"""

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
from Scene.JsonManager import load_json
from Scene.Map import Map
from Scene.Robot import Robot
from Types import State2D


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
    from Types import ConvexShape
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
    
    # Draw robot footprint at start position (optional)
    robot_rect = patches.Rectangle(
        (start[0] - robot.length/2, start[1] - robot.width/2),
        robot.length, robot.width,
        linewidth=1, edgecolor='green', facecolor='none', 
        linestyle='--', alpha=0.5, zorder=4
    )
    ax.add_patch(robot_rect)
    
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
    from Types import ConvexShape
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
    else:# if planner_type == PlannerTypes.RRTStarPlanner:
        rewire_radius_factor = planner_params["rewire_radius_factor"]
        rrt_timeout = planner_params["timeout"]
        planner: Planner = RRTStarPlanner(obstacle_checker=obstacle_checker, world_bounds=world_bounds, timeout=rrt_timeout, rewire_radius_factor=rewire_radius_factor)
    
    # Test cases: (start, goal, description)
    test_cases = [
        (
            (5.0, 5.0, 0.0),
            (map_obj.length - 5.0, map_obj.width - 5.0, 0.0),
            "Corner to opposite corner"
        ),
        (
            (map_obj.length / 2, map_obj.width / 2, 0.0),
            (map_obj.length / 2 + 2.0, map_obj.width / 2 + 2.0, math.pi / 4),
            "Center to nearby position"
        ),
        (
            (5.0, map_obj.width / 2, math.pi / 2),
            (map_obj.length - 5.0, map_obj.width / 2, math.pi / 2),
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
        ((1.0, 1.0, 0.0), "Free space (corner)"),
        ((map_obj.length / 2, map_obj.width / 2, 0.0), "Center"),
        ((0.1, 0.1, 0.0), "Near boundary"),
    ]
    
    print(f"\nTesting {len(test_points)} points for collisions:\n")
    for point, description in test_points:
        is_collision = obstacle_checker.is_collision(point)
        status = "COLLISION" if is_collision else "CLEAR"
        print(f"  {description:30s} -> {status}")


if __name__ == "__main__":
    try:
        test_planner()
        test_collision_detection()
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
