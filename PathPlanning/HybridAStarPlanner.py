from __future__ import annotations

import heapq
import math
from typing import Dict, List, Tuple, Optional

from ObstacleDetection.ObstacleDetector import ObstacleChecker
from PathPlanning.Planners import Node, Planner
from Types import State2D


class HybridAStarPlanner(Planner):
    """
    Hybrid A* path planner for a holonomic robot with (x, y, theta) state space.
    
    Unlike standard A*, Hybrid A* uses:
    - Discrete grid search for coarse planning
    - Continuous motion primitives for smooth paths
    - Treats angle as a continuous dimension for better steering
    """
    
    def __init__(self, obstacle_checker: ObstacleChecker, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 grid_resolution: float = 0.5, angle_bins: int = 16, max_iterations: int = 10000, motion_primitives: Optional[List[Tuple[float, float, float]]] = None) -> None:
        """
        Initialize the Hybrid A* planner.
        
        Args:
            obstacle_checker: Object that checks collisions
            grid_resolution: Discretization of x, y space
            angle_bins: Number of discrete angle bins (coarser = faster)
            max_iterations: Maximum number of nodes to expand
            motion_primitives: List of (dx, dy, dtheta) primitives. If None, use default 16 primitives.
        """
        self.obstacle_checker: ObstacleChecker = obstacle_checker
        self._grid_resolution: float = grid_resolution
        '''The discretization step for x and y.'''
        self._angle_bins: int = angle_bins
        '''The number of discrete bins for theta.'''
        self._angle_step: float = 2 * math.pi / angle_bins
        '''The discretization step for theta.'''
        self._max_iterations: int = max_iterations
        self._closed_set: set[Tuple[int, int, int]] = set()  # (grid_x, grid_y, angle_bin)
        ''' The set of visited discrete states.'''
        self._open_set: List[Node] = []
        '''The priority queue of nodes to expand.'''
        self._g_values: Dict[Tuple[int, int, int], float] = {}
        '''The cost-to-come for each discrete state.'''
        self._world_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = world_bounds
        '''The world boundaries as ((x_min, x_max), (y_min, y_max)).'''
        
        # Define motion primitives for holonomic robot
        # Can move in any direction continuously
        if motion_primitives is None:
            self._motion_primitives = self._generate_default_primitives()
        else:
            self._motion_primitives = motion_primitives
    
    def _generate_default_primitives(self) -> List[Tuple[float, float, float]]:
        """Generate default motion primitives for holonomic system."""
        primitives = []
        
        # 8-directional movement with various speeds and steering angles
        directions = [
            (1, 0),   # Right
            (0, 1),   # Up
            (-1, 0),  # Left
            (0, -1),  # Down
            (1, 1),   # Diagonal up-right
            (-1, 1),  # Diagonal up-left
            (-1, -1), # Diagonal down-left
            (1, -1),  # Diagonal down-right
        ]
        
        base_distance = self._grid_resolution
        
        for dx_dir, dy_dir in directions:
            # Base movement
            dx = dx_dir * base_distance
            dy = dy_dir * base_distance
            
            # No rotation, slight rotation, or opposite steering
            for dtheta in [-self._angle_step, 0, self._angle_step]:
                primitives.append((dx, dy, dtheta))
        
        return primitives
    
    def plan(self, start: State2D, goal: State2D) -> Optional[List[State2D]]:
        """
        Plan a path from start to goal using Hybrid A*.
        
        Args:
            start: Starting state (x, y, theta)
            goal: Goal state (x, y, theta)
            world_bounds: ((x_min, x_max), (y_min, y_max))
            
        Returns:
            List of states representing the path, or None if no path found
        """
        # Check if start and goal are valid
        if self.obstacle_checker.is_collision(start):
            print(f"Start state {start} is in collision!")
            return None
        
        if self.obstacle_checker.is_collision(goal):
            print(f"Goal state {goal} is in collision!")
            return None
        
        # Initialize
        self._closed_set.clear()
        self._open_set.clear()
        self._g_values.clear()
        
        start_node = Node(
            state=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
        )
        heapq.heappush(self._open_set, start_node)
        
        start_key = self._discretize_state(start)
        self._g_values[start_key] = 0.0
        
        iterations = 0
        
        while self._open_set and iterations < self._max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current_node = heapq.heappop(self._open_set)
            current_key = self._discretize_state(current_node.state)
            
            # Skip if already processed
            if current_key in self._closed_set:
                continue
            
            # Check if goal reached
            if self._states_close(current_node.state, goal):
                return self._reconstruct_path(current_node)
            
            # Mark as visited
            self._closed_set.add(current_key)
            
            # Expand using motion primitives
            for next_state, cost in self._expand_node(current_node.state, self._world_bounds):                
                # Check for collisions
                if self.obstacle_checker.is_collision(next_state) or not self.obstacle_checker.is_path_clear(current_node.state, next_state):
                    continue
                
                next_key = self._discretize_state(next_state)
                
                if next_key in self._closed_set:
                    continue
                
                g_cost = current_node.g_cost + cost
                
                # Skip if we've found a better path to this state
                if next_key in self._g_values and g_cost >= self._g_values[next_key]:
                    continue
                
                h_cost = self._heuristic(next_state, goal)
                
                next_node = Node(
                    state=next_state,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current_node,
                )
                
                self._g_values[next_key] = g_cost
                heapq.heappush(self._open_set, next_node)
        
        print(f"No path found after {iterations} iterations")
        return None
    
    def _expand_node(self, state: State2D, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> List[Tuple[State2D, float]]:
        """Expand node using motion primitives."""
        x, y, theta = state
        (x_min, x_max), (y_min, y_max) = bounds
        successors = []
        
        for dx, dy, dtheta in self._motion_primitives:
            new_x = x + dx
            new_y = y + dy
            new_theta = self._normalize_angle(theta + dtheta)
            
            # Check bounds
            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                continue
            
            new_state = (new_x, new_y, new_theta)
            
            # Euclidean cost for motion
            motion_cost = math.sqrt(dx**2 + dy**2)
            
            successors.append((new_state, motion_cost))
        
        return successors
    
    def _discretize_state(self, state: State2D) -> Tuple[int, int, int]:
        """Convert continuous state to discrete grid key for closed set."""
        x, y, theta = state
        grid_x = round(x / self._grid_resolution)
        grid_y = round(y / self._grid_resolution)
        angle_bin = round(theta / self._angle_step) % self._angle_bins
        return (grid_x, grid_y, angle_bin)
    
    def _heuristic(self, state1: State2D, goal: State2D) -> float:
        """
        Heuristic function: Euclidean distance in (x, y).
        For holonomic systems, this is admissible since any (x, y) is reachable.
        """
        x1, y1, _ = state1
        x2, y2, _ = goal
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _states_close(self, state1: State2D, state2: State2D, pos_tol: float = 0.5, angle_tol: Optional[float] = None) -> bool:
        """Check if two states are close enough to consider as goal reached."""
        x1, y1, theta1 = state1
        x2, y2, theta2 = state2
        
        pos_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if angle_tol is None:
            angle_tol = self._angle_step
        
        angle_distance = abs(self._normalize_angle(theta2 - theta1))
        angle_distance = min(angle_distance, 2 * math.pi - angle_distance)
        
        return pos_distance < pos_tol and angle_distance < angle_tol
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _reconstruct_path(self, node: Node) -> List[State2D]:
        """Reconstruct path from start to current node."""
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))
