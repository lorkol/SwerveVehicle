"""
Test file for MPPI Controller
Tests the MPPI controller with path planning trajectories.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple

from Scene.JsonManager import load_json
from Scene.Map import Map
from Scene.Robot import Robot
from PathPlanning.Planners import PlannerTypes
from PathPlanning.AStarPlanner import AStarPlanner
from PathPlanning.RRT_StarPlanner import RRTStarPlanner
from PathPlanning.HybridAStarPlanner import HybridAStarPlanner
from PathPlanning.DStarPlanner import DStarPlanner
from PathPlanning.TrajectoryGenerator import TrajectoryGenerator
from PathController.Robot_Sim import Robot_Sim
from PathController.MPPI.MPPI_Controller import MPPIController
from ActuatorController.ActuatorController import ActuatorController
from ObstacleDetection.ObstacleDetector import StaticObstacleChecker
from Types import State2D, ConvexShape


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
        
        # Extract problem statement
        problem = self.params.get("Problem Statement", {})
        self.start = tuple(problem.get("Start", [10.0, 10.0, 0.0]))
        self.goal = tuple(problem.get("Goal", [90.0, 40.0, 0.0]))
    
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
                step_size=params.get("step_size", 1.0),
                max_iterations=params.get("max_iterations", 10000),
                goal_sample_rate=params.get("goal_sample_rate", 0.1),
                search_radius=params.get("search_radius", 2.0),
                timeout=params.get("timeout", 10.0),
                obstacle_checker=self.obstacle_checker,
                world_bounds=self.world_bounds
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
        
        print(f"✓ Path found with {len(path)} waypoints")
        return path
    
    def generate_trajectory(self, path: List[State2D]) -> np.ndarray:
        """Generate reference trajectory from path."""
        control_config = self.params.get("Control", {})
        mppi_config = control_config.get("MPPI", {})
        
        dt = control_config.get("MPC", {}).get("dt", 0.01)
        horizon = mppi_config.get("N_Horizon", 5)
        max_velocity = 2.0  # m/s
        
        traj_gen = TrajectoryGenerator(dt=dt, horizon=horizon, max_velocity=max_velocity)
        
        # Initial state: start position with zero velocity
        current_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print(f"\nGenerating reference trajectory (horizon={horizon}, dt={dt})...")
        ref_traj = traj_gen.get_reference_trajectory(current_state, path)
        
        print(f"✓ Reference trajectory shape: {ref_traj.shape}")
        return ref_traj
    
    def simulate_controller(self, ref_traj: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Simulate MPPI controller following reference trajectory."""
        control_config = self.params.get("Control", {})
        mppi_config = control_config.get("MPPI", {})
        dt = control_config.get("MPC", {}).get("dt", 0.01)
        
        # Create robot simulator
        robot_sim = Robot_Sim(self.actuator_controller, self.robot)
        
        # Initial state
        initial_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot_sim.set_state(initial_state)
        
        # Create MPPI controller
        print(f"\nCreating MPPI controller...")
        mppi_controller = MPPIController(
            desired_traj=ref_traj,
            robot_sim=robot_sim,
            collision_check_method=lambda state: self.obstacle_checker.is_collision(state)
        )
        
        # Simulate
        print(f"Simulating controller...")
        executed_states = [initial_state.copy()]
        executed_controls = []
        
        max_steps = 100  # Limit simulation steps
        goal_threshold = 1.0  # Distance to goal
        
        for step in range(max_steps):
            current_state = executed_states[-1]
            
            # Check if reached goal
            dist_to_goal = np.linalg.norm(current_state[:2] - np.array(self.goal[:2]))
            if dist_to_goal < goal_threshold:
                print(f"✓ Reached goal at step {step}")
                break
            
            # Get control command from MPPI
            try:
                control = mppi_controller.get_command(current_state)
                executed_controls.append(control)
            except Exception as e:
                print(f"❌ Error getting control at step {step}: {e}")
                break
            
            # Propagate state
            try:
                next_state = robot_sim.propagate(current_state, control)
                
                # Check collision
                if self.obstacle_checker.is_collision(tuple(next_state[:3])):
                    print(f"❌ Collision detected at step {step}")
                    break
                
                executed_states.append(next_state)
                
                if (step + 1) % 10 == 0:
                    print(f"  Step {step + 1}: pos=({next_state[0]:.2f}, {next_state[1]:.2f}), dist_to_goal={dist_to_goal:.2f}m")
            
            except Exception as e:
                print(f"❌ Error propagating state at step {step}: {e}")
                break
        
        print(f"✓ Simulation completed: {len(executed_states)} states")
        return executed_states, executed_controls
    
    def visualize(self, path: List[State2D], executed_states: List[np.ndarray], ref_traj: np.ndarray):
        """Visualize planning and control results."""
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


if __name__ == "__main__":
    tester = ControllerTester()
    tester.run()
