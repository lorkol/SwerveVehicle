from __future__ import annotations

import heapq
import math
from typing import List, Tuple, Optional


from ObstacleDetection.ObstacleDetector import ObstacleChecker
from PathPlanning.Planners import Node, Planner
from Types import State2D


class AStarPlanner(Planner):
    """A* path planner for a robot with (x, y, theta) configuration space."""
    
    def __init__(self, obstacle_checker: ObstacleChecker, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 grid_resolution: float = 1.0, angle_resolution: float = math.pi / 4, max_iterations: int = None) -> None:
        """
        Initialize the A* planner.
        
        Args:
            obstacle_checker: Object that checks collisions
            grid_resolution: Discretization of x, y space
            angle_resolution: Discretization of theta
            max_iterations: Maximum number of nodes to expand. If None, automatically calculated as grid cells * angle states.
        """
        self.obstacle_checker: ObstacleChecker = obstacle_checker
        self._grid_resolution: float = grid_resolution
        '''The discretization step for x and y.'''
        self._angle_resolution: float = angle_resolution
        '''The discretization step for theta.'''
        self._world_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = world_bounds
        '''The world boundaries as ((x_min, x_max), (y_min, y_max)).'''
        
        # Calculate max_iterations automatically if not provided
        if max_iterations is None:
            x_min, x_max = world_bounds[0]
            y_min, y_max = world_bounds[1]
            x_cells = int(math.ceil((x_max - x_min) / grid_resolution))
            y_cells = int(math.ceil((y_max - y_min) / grid_resolution))
            angle_states = int(math.ceil(2 * math.pi / angle_resolution))
            max_iterations = x_cells * y_cells * angle_states
        
        self._max_iterations: int = max_iterations
        print(f"    [A* Debug] Grid resolution: {grid_resolution}, Angle resolution: {math.degrees(angle_resolution):.1f}°")
        print(f"    [A* Debug] Max iterations set to {self._max_iterations}")
        '''The maximum number of nodes to expand.'''
        self._closed_set: set[State2D] = set()
        '''The set of visited states.'''
        self._open_set: list[Node] = []
        '''The open set priority queue.'''
        self._open_set_states: dict[State2D, float] = {}
        '''Maps states to their f_cost for O(1) lookup.'''
        
    def plan(self, start: State2D, goal: State2D) -> Optional[List[State2D]]:
        """
        Plan a path from start to goal using A*.
        
        Args:
            start: Starting state (x, y, theta)
            goal: Goal state (x, y, theta)
            world_bounds: ((x_min, x_max), (y_min, y_max))
            
        Returns:
            List of states representing the path, or None if no path found
        """
        # Quantize start and goal to grid to ensure they're on the discretized space
        start = self._quantize_state(start)
        goal = self._quantize_state(goal)
        
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
        self._open_set_states.clear()
        
        start_node = Node(
            state=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
        )
        heapq.heappush(self._open_set, start_node)
        self._open_set_states[start] = start_node.f_cost
        
        iterations = 0
        debug_interval = 1000
        
        while self._open_set and iterations < self._max_iterations:
            iterations += 1
            
            # Debug output every N iterations
            if iterations % debug_interval == 0:
                print(f"    [A* Debug] Iteration {iterations}: {len(self._open_set)} nodes in open set, {len(self._closed_set)} nodes in closed set")
            
            # Get node with lowest f_cost
            current_node = heapq.heappop(self._open_set)
            
            # Skip if this node was already processed
            if current_node.state in self._closed_set:
                continue
            
            # Check if goal reached
            if self._states_close(current_node.state, goal):
                print(f"    [A* Debug] Goal reached at iteration {iterations}")
                return self._reconstruct_path(current_node)
            
            # Mark as visited
            self._closed_set.add(current_node.state)
            self._open_set_states.pop(current_node.state, None)
            
            # Expand neighbors
            for next_state, cost in self._get_neighbors(current_node.state):
                if next_state in self._closed_set:
                    continue
                
                g_cost = current_node.g_cost + cost
                h_cost = self._heuristic(next_state, goal)
                
                # Only add if this is a better path than previously found
                if next_state in self._open_set_states and self._open_set_states[next_state] <= g_cost + h_cost:
                    continue
                
                next_node = Node(
                    state=next_state,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current_node,
                )
                
                heapq.heappush(self._open_set, next_node)
                self._open_set_states[next_state] = next_node.f_cost
        
        print(f"    [A* Debug] No path found after {iterations} iterations (max: {self._max_iterations})")
        return None
    
    def _quantize_state(self, state: State2D) -> State2D:
        """
        Quantize state to grid to avoid floating point precision issues.
        Ensures that the same logical grid cell always maps to the same state tuple.
        """
        x, y, theta = state
        # Snap to nearest grid point
        x_quant = round(x / self._grid_resolution) * self._grid_resolution
        y_quant = round(y / self._grid_resolution) * self._grid_resolution
        # Snap angle to nearest angle step
        theta_quant = round(theta / self._angle_resolution) * self._angle_resolution
        theta_quant = self._normalize_angle(theta_quant)
        return (x_quant, y_quant, theta_quant)
    
    def _get_neighbors(self, state: State2D) -> List[Tuple[State2D, float]]:
        """Generate neighboring states from current state."""
        x, y, theta = state
        (x_min, x_max), (y_min, y_max) = self._world_bounds
        neighbors = []
        
        # Generate 8-directional motion in (x, y) with angle variations
        dx_options = [-self._grid_resolution, 0, self._grid_resolution]
        dy_options = [-self._grid_resolution, 0, self._grid_resolution]
        dtheta_options = [-self._angle_resolution, 0, self._angle_resolution]
        
        for dx in dx_options:
            for dy in dy_options:
                for dtheta in dtheta_options:
                    new_x = x + dx
                    new_y = y + dy
                    new_theta = theta + dtheta
                    
                    # Keep theta in [-pi, pi]
                    new_theta = self._normalize_angle(new_theta)
                    
                    # Quantize to grid to prevent floating point precision issues
                    new_state = self._quantize_state((new_x, new_y, new_theta))
                    
                    # Check bounds
                    if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                        continue
                    
                    # Check collisions at new state
                    if self.obstacle_checker.is_collision(new_state):
                        continue
                    
                    # Check if path between states is clear
                    if not self.obstacle_checker.is_path_clear(state, new_state):
                        continue
                                        
                    # Euclidean cost for motion
                    motion_cost = math.sqrt(dx**2 + dy**2)
                    
                    neighbors.append((new_state, motion_cost))
        
        return neighbors
    
    def _heuristic(self, state1: State2D, state2: State2D) -> float:
        """
        Heuristic function: Euclidean distance in (x, y).
        Can be improved with better heuristics.
        """
        x1, y1, _ = state1
        x2, y2, _ = state2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _states_close(self, state1: State2D, state2: State2D) -> bool:
        """Check if two states are close enough (same grid cell)."""
        # Since both states are quantized, they should be exactly equal if on same cell
        # Use small epsilon for floating point comparison
        x1, y1, theta1 = state1
        x2, y2, theta2 = state2
        
        eps = 1e-6  # Very small tolerance for floating point
        return (abs(x2 - x1) < eps and abs(y2 - y1) < eps and 
                abs(self._normalize_angle(theta2 - theta1)) < eps)
    
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