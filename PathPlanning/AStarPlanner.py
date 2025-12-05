from __future__ import annotations

import heapq
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ObstacleDetection.ObstacleDetector import ObstacleChecker
from PathPlanning.Planners import Node, Planner
from Types import State


class AStarPlanner(Planner):
    """A* path planner for a robot with (x, y, theta) configuration space."""
    
    def __init__(self, obstacle_checker: ObstacleChecker, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 grid_resolution: float = 0.5, angle_resolution: float = math.pi / 8, max_iterations: int = 10000) -> None:
        """
        Initialize the A* planner.
        
        Args:
            obstacle_checker: Object that checks collisions
            grid_resolution: Discretization of x, y space
            angle_resolution: Discretization of theta
            max_iterations: Maximum number of nodes to expand
        """
        self.obstacle_checker: ObstacleChecker = obstacle_checker
        self.grid_resolution: float = grid_resolution
        '''The discretization step for x and y.'''
        self.angle_resolution: float = angle_resolution
        '''The discretization step for theta.'''
        self.max_iterations: int = max_iterations
        '''The maximum number of nodes to expand.'''
        self.closed_set: set[State] = set()
        '''The set of visited states.'''
        self.open_set: list[Node] = []
        '''The set of visited states.'''
        self.world_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = world_bounds
        '''The world boundaries as ((x_min, x_max), (y_min, y_max)).'''
        
    def plan(self, start: State, goal: State) -> Optional[List[State]]:
        """
        Plan a path from start to goal using A*.
        
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
        self.closed_set.clear()
        self.open_set.clear()
        
        start_node = Node(
            state=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
        )
        heapq.heappush(self.open_set, start_node)
        
        iterations = 0
        
        while self.open_set and iterations < self.max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current_node = heapq.heappop(self.open_set)
            
            # Check if goal reached
            if self._states_close(current_node.state, goal):
                return self._reconstruct_path(current_node)
            
            # Mark as visited
            self.closed_set.add(current_node.state)
            
            # Expand neighbors
            for next_state, cost in self._get_neighbors(current_node.state, self.world_bounds):
                if next_state in self.closed_set:
                    continue
                
                # Check for collisions
                if self.obstacle_checker.is_collision(next_state):
                    continue
                
                if not self.obstacle_checker.is_path_clear(current_node.state, next_state):
                    continue
                
                g_cost = current_node.g_cost + cost
                h_cost = self._heuristic(next_state, goal)
                
                next_node = Node(
                    state=next_state,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current_node,
                )
                
                heapq.heappush(self.open_set, next_node)
        
        print(f"No path found after {iterations} iterations")
        return None
    
    def _get_neighbors(self, state: State, bounds: Tuple[Tuple[float, float], Tuple[float, float]]) -> List[Tuple[State, float]]:
        """Generate neighboring states from current state."""
        x, y, theta = state
        (x_min, x_max), (y_min, y_max) = bounds
        neighbors = []
        
        # Generate 8-directional motion in (x, y) with angle variations
        dx_options = [-self.grid_resolution, 0, self.grid_resolution]
        dy_options = [-self.grid_resolution, 0, self.grid_resolution]
        dtheta_options = [-self.angle_resolution, 0, self.angle_resolution]
        
        for dx in dx_options:
            for dy in dy_options:
                # Skip staying in place
                if dx == 0 and dy == 0:
                    continue
                
                for dtheta in dtheta_options:
                    new_x = x + dx
                    new_y = y + dy
                    new_theta = theta + dtheta
                    
                    # Keep theta in [-pi, pi]
                    new_theta = self._normalize_angle(new_theta)
                    
                    # Check bounds
                    if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                        continue
                    
                    new_state = (new_x, new_y, new_theta)
                    
                    # Euclidean cost for motion
                    motion_cost = math.sqrt(dx**2 + dy**2)
                    
                    neighbors.append((new_state, motion_cost))
        
        return neighbors
    
    def _heuristic(self, state1: State, state2: State) -> float:
        """
        Heuristic function: Euclidean distance in (x, y).
        Can be improved with better heuristics.
        """
        x1, y1, _ = state1
        x2, y2, _ = state2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _states_close(self, state1: State, state2: State, tol: float = 0.5) -> bool:
        """Check if two states are close enough to consider as goal reached."""
        x1, y1, theta1 = state1
        x2, y2, theta2 = state2
        
        pos_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle_distance = abs(self._normalize_angle(theta2 - theta1))
        
        return pos_distance < tol and angle_distance < self.angle_resolution
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _reconstruct_path(self, node: Node) -> List[State]:
        """Reconstruct path from start to current node."""
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))
