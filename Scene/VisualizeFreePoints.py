"""
Visualize configuration map and generate maximally-spaced free points.
Uses greedy furthest-point sampling to spread points as far apart as possible.
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work from Scene folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from JsonManager import load_json
from Map import Map
from Robot import Robot
from Types import ConvexShape
from ObstacleDetection.ObstacleDetector import StaticObstacleChecker


class FreePointFinder:
    """Find maximally-spaced free points in a map."""
    
    def __init__(self, map_obj: Map, robot: Robot, obstacles, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        """
        Initialize the point finder.
        
        Args:
            map_obj: Map object
            robot: Robot object
            obstacles: List of obstacles
            world_bounds: ((x_min, x_max), (y_min, y_max))
        """
        self.map_obj = map_obj
        self.robot = robot
        self.obstacles = obstacles
        self.world_bounds = world_bounds
        self.obstacle_checker = StaticObstacleChecker(
            robot=robot,
            obstacles=obstacles,
            map_limits=world_bounds,
            use_parallelization=False
        )
    
    def is_point_free(self, x: float, y: float, theta: float = 0.0) -> bool:
        """Check if a point is collision-free."""
        state = (x, y, theta)
        return not self.obstacle_checker.is_collision(state)
    
    def find_free_points(self, num_points: int = 10, num_candidates: int = 5000, 
                        min_distance: float = 1.0, max_attempts: int = 100) -> List[Tuple[float, float]]:
        """
        Find maximally-spaced free points using greedy furthest-point sampling.
        
        Args:
            num_points: Number of points to find
            num_candidates: Number of candidate points to sample (higher = better spacing, slower)
            min_distance: Minimum distance from obstacles/boundaries
            max_attempts: Maximum attempts to find valid points
            
        Returns:
            List of (x, y) points
        """
        (x_min, x_max), (y_min, y_max) = self.world_bounds
        
        # Generate candidate pool
        print(f"Generating {num_candidates} candidate points...")
        candidates = []
        attempts = 0
        
        while len(candidates) < num_candidates and attempts < max_attempts * 10:
            x = random.uniform(x_min + min_distance, x_max - min_distance)
            y = random.uniform(y_min + min_distance, y_max - min_distance)
            
            if self.is_point_free(x, y):
                candidates.append((x, y))
            
            attempts += 1
        
        print(f"  Generated {len(candidates)} valid candidate points")
        
        if len(candidates) < num_points:
            print(f"  Warning: Only found {len(candidates)} free points, need {num_points}")
            return candidates
        
        # Greedy furthest-point sampling
        print(f"Finding {num_points} maximally-spaced points...")
        selected_points = []
        
        # Start with random point
        start_idx = random.randint(0, len(candidates) - 1)
        selected_points.append(candidates[start_idx])
        remaining_indices = set(range(len(candidates)))
        remaining_indices.discard(start_idx)
        
        # Iteratively add furthest point
        for iteration in range(1, num_points):
            if not remaining_indices:
                print(f"  Warning: Ran out of candidates at {iteration} points")
                break
            
            # Find candidate with maximum minimum distance to selected points
            best_idx = None
            best_min_distance = -1
            
            for idx in remaining_indices:
                candidate = candidates[idx]
                min_dist_to_selected = min(
                    self._distance(candidate, selected) 
                    for selected in selected_points
                )
                
                if min_dist_to_selected > best_min_distance:
                    best_min_distance = min_dist_to_selected
                    best_idx = idx
            
            if best_idx is None:
                break
            
            selected_points.append(candidates[best_idx])
            remaining_indices.discard(best_idx)
            
            if (iteration) % max(1, num_points // 5) == 0:
                print(f"  Found {iteration} points (min spacing: {best_min_distance:.2f}m)")
        
        # Calculate statistics
        if len(selected_points) > 1:
            min_spacing = min(
                self._distance(selected_points[i], selected_points[j])
                for i in range(len(selected_points))
                for j in range(i + 1, len(selected_points))
            )
            avg_spacing = np.mean([
                self._distance(selected_points[i], selected_points[j])
                for i in range(len(selected_points))
                for j in range(i + 1, len(selected_points))
            ])
            print(f"\n✓ Found {len(selected_points)} points")
            print(f"  Minimum spacing: {min_spacing:.2f}m")
            print(f"  Average spacing: {avg_spacing:.2f}m")
        
        return selected_points
    
    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def visualize(self, free_points: List[Tuple[float, float]], title: str = "Free Points Visualization") -> None:
        """
        Visualize the map with free points.
        
        Args:
            free_points: List of (x, y) points to visualize
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        (x_min, x_max), (y_min, y_max) = self.world_bounds
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect('equal')
        
        # Draw map boundary
        boundary = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='lightgray', 
            alpha=0.1, label='Map Boundary'
        )
        ax.add_patch(boundary)
        
        # Draw grid
        grid_spacing = 10.0
        for x in range(0, int(x_max) + 1, int(grid_spacing)):
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
        for y in range(0, int(y_max) + 1, int(grid_spacing)):
            ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
        
        # Draw obstacles
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.shape == ConvexShape.Circle:
                circle = patches.Circle(
                    (obstacle.center[0], obstacle.center[1]),
                    obstacle.radius,
                    linewidth=2, edgecolor='red', facecolor='red', 
                    alpha=0.4, label='Obstacles' if i == 0 else ''
                )
                ax.add_patch(circle)
            
            elif obstacle.shape == ConvexShape.Polygon:
                polygon = patches.Polygon(
                    obstacle.points,
                    linewidth=2, edgecolor='red', facecolor='red', 
                    alpha=0.4, label='Obstacles' if i == 0 else ''
                )
                ax.add_patch(polygon)
        
        # Draw free points
        if free_points:
            points_x = [p[0] for p in free_points]
            points_y = [p[1] for p in free_points]
            ax.scatter(points_x, points_y, s=100, c='green', marker='o', 
                      edgecolors='darkgreen', linewidth=2, label='Free Points', zorder=10)
            
            # Add point numbers
            for i, (x, y) in enumerate(free_points):
                ax.text(x, y + 0.5, f'{i+1}', fontsize=10, ha='center', 
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            # Draw connections showing spacing
            for i in range(len(free_points)):
                for j in range(i + 1, len(free_points)):
                    p1, p2 = free_points[i], free_points[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', alpha=0.2, linewidth=0.5)
        
        # Labels and legend
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f"{title} ({len(free_points)} points)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function."""
    # Load configuration
    print("Loading configuration...")
    config_path = "Scene/Configuration.json"
    config = load_json(config_path)
    
    # Extract robot and map config
    robot_config = config.get("Robot", {})
    map_config = config.get("Map", {})
    
    # Create objects
    robot = Robot(robot_config)
    map_obj = Map(map_config)
    
    # Define world bounds
    world_bounds = ((0, map_obj.length), (0, map_obj.width))
    
    print(f"Map dimensions: {map_obj.length} x {map_obj.width}")
    print(f"Number of obstacles: {len(map_obj.obstacles)}")
    
    # Find free points
    finder = FreePointFinder(map_obj, robot, map_obj.obstacles, world_bounds)
    
    # Find points with defaults
    free_points = finder.find_free_points(num_points=10, num_candidates=5000)
    
    # Display points
    print("\nFree Points Coordinates:")
    for i, (x, y) in enumerate(free_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")
    
    # Visualize
    finder.visualize(free_points)


if __name__ == "__main__":
    main()
