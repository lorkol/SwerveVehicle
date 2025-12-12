"""
D* (Dynamic A*) Path Planner
Efficient replanning algorithm for dynamic environments with obstacle changes.
"""

from __future__ import annotations

import math
import heapq
from typing import List, Tuple, Optional, Dict

from ObstacleDetection.ObstacleDetector import ObstacleChecker
from PathPlanning.Planners import Node, Planner
from Types import State2D


class DStarPlanner(Planner):
    """D* path planner for a robot with (x, y, theta) configuration space."""
    
    def __init__(self, obstacle_checker: ObstacleChecker, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 grid_resolution: float = 1.0, angle_resolution: float = math.pi / 4, max_iterations: Optional[int] = None) -> None:
        """
        Initialize the D* planner.
        
        Args:
            obstacle_checker: Object that checks collisions
            world_bounds: ((x_min, x_max), (y_min, y_max))
            grid_resolution: Resolution of the configuration space grid
            angle_resolution: Resolution for angle discretization
            max_iterations: Maximum iterations (auto-calculated if None)
        """
        self._obstacle_checker: ObstacleChecker = obstacle_checker
        self._world_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = world_bounds
        '''The world boundaries as ((x_min, x_max), (y_min, y_max)).'''
        
        self._grid_resolution: float = grid_resolution
        '''Grid resolution for discretizing the configuration space.'''
        
        self._angle_resolution: float = angle_resolution
        '''Angular resolution for discretizing orientation.'''
        
        # Auto-calculate max_iterations based on grid size
        x_min, x_max = world_bounds[0]
        y_min, y_max = world_bounds[1]
        x_cells = max(1, int((x_max - x_min) / grid_resolution))
        y_cells = max(1, int((y_max - y_min) / grid_resolution))
        angle_cells = max(1, int(2 * math.pi / angle_resolution))
        
        if max_iterations is None:
            max_iterations = x_cells * y_cells * angle_cells
        
        self._max_iterations: int = max_iterations
        '''Maximum iterations for planning.'''
        
        self._open_set: List[Tuple[float, State2D]] = []
        '''Priority queue of states to expand (k_min, state).'''
        
        self._g_costs: Dict[State2D, float] = {}
        '''Cost from goal to each state.'''
        
        self._rhs_costs: Dict[State2D, float] = {}
        '''Right-hand side (lookahead) cost for each state.'''
        
        self._came_from: Dict[State2D, Optional[State2D]] = {}
        '''Parent pointers for path reconstruction.'''
        
        self._k_min: float = 0.0
        '''Minimum key value in open set.'''
    
    def plan(self, start: State2D, goal: State2D) -> Optional[List[State2D]]:
        """
        Plan a path from start to goal using D*.
        
        Args:
            start: Starting state (x, y, theta)
            goal: Goal state (x, y, theta)
            
        Returns:
            List of states representing the path, or None if no path found
        """
        # Validate start and goal
        if self._obstacle_checker.is_collision(start):
            print(f"Start state {start} is in collision!")
            return None
        
        if self._obstacle_checker.is_collision(goal):
            print(f"Goal state {goal} is in collision!")
            return None
        
        # Initialize
        self._open_set.clear()
        self._g_costs.clear()
        self._rhs_costs.clear()
        self._came_from.clear()
        self._k_min = 0.0
        
        # Initialize goal state
        self._g_costs[goal] = float('inf')
        self._rhs_costs[goal] = 0.0
        
        self._insert(goal, 0.0)
        
        iterations = 0
        debug_interval = 1000
        
        while self._open_set and iterations < self._max_iterations:
            iterations += 1
            
            # Debug output
            if iterations % debug_interval == 0:
                print(f"    [D* Debug] Iteration {iterations}: {len(self._open_set)} open nodes, g({start}) = {self._g_costs.get(start, float('inf')):.2f}")
            
            # Get node with minimum key
            k_old, current = self._top_key()
            
            # Expand node
            if current == start:
                # Reached start state with consistent costs
                if abs(self._g_costs.get(start, float('inf')) - self._rhs_costs.get(start, float('inf'))) < 1e-6:
                    # Path found
                    path = self._reconstruct_path(start, goal)
                    if path:
                        print(f"    [D* Debug] Path found at iteration {iterations}")
                        return path
            
            # Remove from open set
            self._open_set.pop(0)
            
            if k_old < self._calculate_key(current)[0]:
                # Inconsistent - insert again with new key
                self._insert(current, self._calculate_key(current)[0])
            elif self._g_costs.get(current, float('inf')) > self._rhs_costs.get(current, float('inf')):
                # Overconsistent
                self._g_costs[current] = self._rhs_costs.get(current, float('inf'))
                
                # Update predecessors (states that can reach current)
                for neighbor in self._get_neighbors(current):
                    self._update_vertex(neighbor)
            else:
                # Underconsistent
                old_g = self._g_costs.get(current, float('inf'))
                self._g_costs[current] = float('inf')
                
                # Update current and predecessors
                self._update_vertex(current)
                for neighbor in self._get_neighbors(current):
                    self._update_vertex(neighbor)
        
        print(f"    [D* Debug] Planning completed after {iterations} iterations (max: {self._max_iterations})")
        
        if start in self._g_costs and self._g_costs[start] < float('inf'):
            print(f"    [D* Debug] Partial path found with cost {self._g_costs[start]:.2f}")
            return self._reconstruct_path(start, goal)
        else:
            print(f"    [D* Debug] No path found")
            return None
    
    def _update_vertex(self, state: State2D) -> None:
        """Update the cost of a vertex."""
        if state not in self._g_costs:
            self._g_costs[state] = float('inf')
        if state not in self._rhs_costs:
            self._rhs_costs[state] = float('inf')
        
        goal = list(self._rhs_costs.keys())[0] if self._rhs_costs else state
        
        # Calculate RHS: minimum over successors
        if state == goal:
            rhs = 0.0
        else:
            rhs = float('inf')
            for neighbor in self._get_neighbors(state):
                cost = self._g_costs.get(neighbor, float('inf')) + self._distance(state, neighbor)
                if cost < rhs:
                    rhs = cost
                    self._came_from[state] = neighbor
        
        self._rhs_costs[state] = rhs
        
        # Update open set
        if abs(self._g_costs.get(state, float('inf')) - rhs) > 1e-6:
            self._insert(state, self._calculate_key(state)[0])
        else:
            # Remove from open if consistent
            self._open_set = [(k, s) for k, s in self._open_set if s != state]
            heapq.heapify(self._open_set)
    
    def _calculate_key(self, state: State2D) -> Tuple[float, float]:
        """Calculate key (priority) for a state."""
        g = self._g_costs.get(state, float('inf'))
        rhs = self._rhs_costs.get(state, float('inf'))
        h = self._heuristic(state, list(self._rhs_costs.keys())[0] if self._rhs_costs else state)
        
        k1 = min(g, rhs) + h
        k2 = min(g, rhs)
        
        return (k1, k2)
    
    def _insert(self, state: State2D, key: float) -> None:
        """Insert or update a state in the open set."""
        # Remove if already present
        self._open_set = [(k, s) for k, s in self._open_set if s != state]
        
        # Insert with new key
        heapq.heappush(self._open_set, (key, state))
    
    def _top_key(self) -> Tuple[float, State2D]:
        """Get the state with minimum key from open set."""
        if not self._open_set:
            return (float('inf'), (0.0, 0.0, 0.0))
        
        key, state = self._open_set[0]
        return (key, state)
    
    def _get_neighbors(self, state: State2D) -> List[State2D]:
        """Get all valid neighboring states."""
        x, y, theta = state
        neighbors = []
        
        # Generate 27 neighbors (3x3x3 motion options)
        for dx in [-self._grid_resolution, 0, self._grid_resolution]:
            for dy in [-self._grid_resolution, 0, self._grid_resolution]:
                for dtheta in [-self._angle_resolution, 0, self._angle_resolution]:
                    if dx == 0 and dy == 0 and dtheta == 0:
                        continue
                    
                    new_x = x + dx
                    new_y = y + dy
                    new_theta = self._normalize_angle(theta + dtheta)
                    
                    # Check bounds
                    x_min, x_max = self._world_bounds[0]
                    y_min, y_max = self._world_bounds[1]
                    
                    if new_x < x_min or new_x > x_max or new_y < y_min or new_y > y_max:
                        continue
                    
                    neighbor = (new_x, new_y, new_theta)
                    
                    # Check collision
                    if not self._obstacle_checker.is_collision(neighbor):
                        if self._obstacle_checker.is_path_clear(state, neighbor):
                            neighbors.append(neighbor)
        
        return neighbors
    
    def _distance(self, state1: State2D, state2: State2D) -> float:
        """Calculate Euclidean distance between two states (ignoring theta)."""
        x1, y1, _ = state1
        x2, y2, _ = state2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _heuristic(self, state1: State2D, state2: State2D) -> float:
        """Heuristic: Euclidean distance in (x, y)."""
        return self._distance(state1, state2)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _reconstruct_path(self, start: State2D, goal: State2D) -> Optional[List[State2D]]:
        """Reconstruct path from start to goal."""
        path = [start]
        current = start
        visited = set()
        max_steps = 10000
        
        while current != goal and len(path) < max_steps:
            if current in visited:
                # Cycle detected
                return None
            
            visited.add(current)
            
            # Find neighbor with lowest cost to goal
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in self._get_neighbors(current):
                cost = self._g_costs.get(neighbor, float('inf')) + self._distance(current, neighbor)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                return None if len(path) == 1 else path
            
            # Check if we're getting closer to goal
            current_dist = self._distance(current, goal)
            neighbor_dist = self._distance(best_neighbor, goal)
            
            if neighbor_dist >= current_dist and len(path) > 1:
                # Not making progress
                return path
            
            path.append(best_neighbor)
            current = best_neighbor
        
        return path if current == goal else None
