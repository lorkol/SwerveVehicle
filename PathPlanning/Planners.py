from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from ObstacleDetection.ObstacleDetector import ObstacleChecker
from Types import State2D
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev


class PlannerTypes(Enum):
    AStarPlanner = "AStarPlanner"
    HybridAStarPlanner = "HybridAStarPlanner"
    RRTStarPlanner = "RRTStarPlanner"
    
class Planner(ABC):
    """Abstract base class for path planners."""
    
    @abstractmethod
    def plan(self, start: State2D, goal: State2D) -> Optional[List[State2D]]:
        """Plan a path from start to goal. Return list of states or None if no path found."""
        pass
    
    

@dataclass
class Node:
    """Represents a state node in the Hybrid A* search tree."""
    state: State2D  # (x, y, theta)
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Optional["Node"] = None
    
    @property
    def f_cost(self) -> float:
        """Total estimated cost: g + h"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: "Node") -> bool:
        """For heap ordering - lower f_cost has priority."""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.state == other.state


def smooth_path(path: List[State2D], obstacle_checker: ObstacleChecker, downsample_factor: int = 5) -> List[State2D]:
    """
    Smooth the path using B-Spline and then downsample for controller use.
    Tries multiple smoothing strategies, progressively less aggressive if earlier ones fail.
    Validates collision-free before accepting a smoothing strategy.
    
    Args:
        path: Original path from planner
        obstacle_checker: collision checker to validate smoothed segments
        downsample_factor: Reduce waypoints by this factor (2 = every other point)
    
    Returns:
        Smoothed and downsampled path safe for controller
    """
    if len(path) < 4:
        return path
    
    # Remove duplicate/very close points first
    unique_points = []
    for point in path:
        if not unique_points or (abs(point[0] - unique_points[-1][0]) > 1e-6 or 
                                 abs(point[1] - unique_points[-1][1]) > 1e-6):
            unique_points.append(point)
    
    if len(unique_points) < 4:
        return path
    
    x = [p[0] for p in unique_points]
    y = [p[1] for p in unique_points]
    
    # Try different smoothing strategies in order of aggressiveness
    smoothing_strategies = [
        # (k_degree, s_smoothing, description)
        (3, 10.0, "cubic with ultra-aggressive smoothing"),
        (3, 5.0, "cubic with aggressive smoothing"),
        (3, 2.0, "cubic with moderate smoothing"),
        (3, 0.5, "cubic with light smoothing"),
        (2, 2.0, "quadratic with moderate smoothing"),
        (2, 0.5, "quadratic with light smoothing"),
        (1, 0.0, "linear interpolation"),
    ]
    
    smoothed = None
    successful_strategy = None
    
    for k, s, description in smoothing_strategies:
        try:
            # Ensure we have enough points for this spline degree
            if len(unique_points) < k + 1:
                continue
            
            print(f"Trying {description}...")
            
            # Fit B-Spline
            if s > 0:
                tck, u = splprep([x, y], s=s, k=k)
            else:
                # s=0 means pass through all points exactly
                tck, u = splprep([x, y], s=0, k=k)
            
            # Generate finer resolution path
            u_new: np.ndarray = np.linspace(0, 1, num=len(unique_points) * 5)
            new_x, new_y = splev(u_new, tck)
            
            # Re-calculate theta based on smooth derivatives
            dx, dy = splev(u_new, tck, der=1)
            new_theta: np.ndarray = np.arctan2(dy, dx)
            
            smoothed_path: List[State2D] = []
            for i in range(len(new_x)):
                smoothed_path.append((new_x[i], new_y[i], new_theta[i])) # type: ignore
            
            # Check collisions before accepting this strategy
            has_collision = False
            for i in range(len(smoothed_path) - 1):
                state1 = smoothed_path[i]
                state2 = smoothed_path[i + 1]
                
                # Check if path segment is collision-free
                if not obstacle_checker.is_path_clear(state1, state2):
                    has_collision = True
                    break
                
                # Check if new state is collision-free
                if obstacle_checker.is_collision(state2):
                    has_collision = True
                    break
            
            if has_collision:
                print(f"  ✗ Collision detected in smoothed path with {description}, trying next strategy...")
                continue
            
            print(f"  ✓ Path smoothing successful using {description}, no collisions detected")
            smoothed = smoothed_path
            successful_strategy = description
            break
        
        except Exception as e:
            print(f"  ✗ Path smoothing failed with {description}: {e}, trying next strategy...")
            continue
    
    # If all smoothing strategies fail, use original path
    if smoothed is None:
        print(f"All smoothing strategies failed or had collisions, using original path with {len(path)} waypoints")
        smoothed = path
    
    # Step 2: Downsample the path
    downsampled = smoothed[::downsample_factor]
    
    # Always include the goal
    if downsampled[-1] != smoothed[-1]:
        downsampled.append(smoothed[-1])
    
    # Step 3: Final validation of downsampled path
    has_collision = False
    for i in range(len(downsampled) - 1):
        state1 = downsampled[i]
        state2 = downsampled[i + 1]
        
        # Check if path segment is collision-free
        if not obstacle_checker.is_path_clear(state1, state2):
            print(f"Warning: Downsampled segment from {state1} to {state2} has collision!")
            has_collision = True
        
        # Check if new state is collision-free
        if obstacle_checker.is_collision(state2):
            print(f"Warning: Downsampled state {state2} is in collision!")
            has_collision = True
    
    if has_collision:
        print("Warning: Downsampled path has collisions, returning original path")
        return path
    
    if successful_strategy:
        print(f"Final path: {len(smoothed)} waypoints smoothed -> {len(downsampled)} waypoints downsampled (factor {downsample_factor})")
    
    return downsampled