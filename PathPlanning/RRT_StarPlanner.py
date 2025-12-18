"""
RRT* (Rapidly-exploring Random Tree Star) Path Planner
Asymptotically optimal path planning algorithm with timeout support.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

import numpy as np

from ObstacleDetection.ObstacleDetector import ObstacleChecker
from PathPlanning.Planners import Node, Planner, theta_smooth_path
from Types import OptionalPathType, PathType, State2D


class RRTStarPlanner(Planner):
    """RRT* path planner for a robot with (x, y, theta) configuration space."""
    
    def __init__(self, obstacle_checker: ObstacleChecker, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]], step_size: float = 1.0, angle_step_size: float = math.pi / 4,
                 timeout: float = 10.0, goal_sample_rate: float = 0.1, rewire_radius_factor: float = 10.0) -> None:
        """
        Initialize the RRT* planner.
        
        Args:
            obstacle_checker: Object that checks collisions
            world_bounds: ((x_min, x_max), (y_min, y_max))
            step_size: Maximum distance to extend tree in each iteration
            angle_step_size: Maximum angle change per step
            timeout: Maximum time in seconds for planning
            goal_sample_rate: Probability of sampling goal state (0-1)
            rewire_radius_factor: Factor for rewiring radius (larger = more rewiring)
        """
        self._obstacle_checker: ObstacleChecker = obstacle_checker
        self._world_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = world_bounds
        '''The world boundaries as ((x_min, x_max), (y_min, y_max)).'''
        
        self._step_size: float = step_size
        '''Maximum distance to extend tree in each iteration.'''
        
        self._angle_step_size: float = angle_step_size
        '''Maximum angle change per step.'''
        
        self._timeout: float = timeout
        '''Maximum time in seconds for planning.'''
        
        self._goal_sample_rate: float = goal_sample_rate
        '''Probability of sampling goal state instead of random state.'''
        
        self._rewire_radius_factor: float = rewire_radius_factor
        '''Factor for rewiring radius calculation.'''
        
        self._tree_nodes: List[Node] = []
        '''All nodes in the RRT* tree.'''
        
        self._best_path: OptionalPathType = None
        '''Best path found so far.'''
        
        self._best_cost: float = float('inf')
        '''Cost of best path found so far.'''
    
    def plan(self, start: State2D, goal: State2D) -> OptionalPathType:
        """
        Plan a path from start to goal using RRT*.
        
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
        self._tree_nodes.clear()
        self._best_path = None
        self._best_cost = float('inf')
        
        start_node = Node(
            state=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal)
        )
        self._tree_nodes.append(start_node)
        
        start_time = time.time()
        iterations = 0
        debug_interval = 1000
        
        while time.time() - start_time < self._timeout:
            iterations += 1
            
            # Debug output
            if iterations % debug_interval == 0:
                elapsed = time.time() - start_time
                print(f"    [RRT* Debug] Iteration {iterations} at {elapsed:.2f}s: {len(self._tree_nodes)} nodes, best cost: {self._best_cost:.2f}")
            
            # Sample random state or goal
            if random.random() < self._goal_sample_rate:
                random_state = goal
            else:
                random_state = self._random_state()
            
            # Find nearest node in tree
            nearest_node = self._nearest_node(random_state)
            
            # Extend towards random state
            new_state = self._steer(nearest_node.state, random_state)
            
            # Check if path is collision-free
            if not self._obstacle_checker.is_path_clear(nearest_node.state, new_state):
                continue
            
            if self._obstacle_checker.is_collision(new_state):
                continue
            
            # Calculate cost
            step_cost = self._distance(nearest_node.state, new_state)
            new_g_cost = nearest_node.g_cost + step_cost
            new_h_cost = self._heuristic(new_state, goal)
            
            # Find nodes within rewiring radius
            rewire_radius = self._calculate_rewire_radius(len(self._tree_nodes))
            nearby_nodes = self._find_nearby_nodes(new_state, rewire_radius)
            
            # Find best parent among nearby nodes
            best_parent = nearest_node
            best_cost = new_g_cost
            
            for nearby_node in nearby_nodes:
                step_cost_nearby = self._distance(nearby_node.state, new_state)
                cost_through_nearby = nearby_node.g_cost + step_cost_nearby
                
                if cost_through_nearby < best_cost:
                    if self._obstacle_checker.is_path_clear(nearby_node.state, new_state):
                        best_parent = nearby_node
                        best_cost = cost_through_nearby
            
            # Create new node
            new_node = Node(
                state=new_state,
                g_cost=best_cost,
                h_cost=new_h_cost,
                parent=best_parent
            )
            self._tree_nodes.append(new_node)
            
            # Rewire: Check if new node can improve nearby nodes
            for nearby_node in nearby_nodes:
                step_cost_to_nearby = self._distance(new_state, nearby_node.state)
                cost_through_new = new_node.g_cost + step_cost_to_nearby
                
                if cost_through_new < nearby_node.g_cost:
                    if self._obstacle_checker.is_path_clear(new_state, nearby_node.state):
                        nearby_node.parent = new_node
                        nearby_node.g_cost = cost_through_new
            
            # Check if goal is reached
            dist_to_goal = self._distance(new_state, goal)
            if dist_to_goal < 0.5 and self._obstacle_checker.is_path_clear(new_state, goal):
                goal_cost = new_node.g_cost + dist_to_goal
                if goal_cost < self._best_cost:
                    self._best_cost = goal_cost
                    # Create virtual goal node for path reconstruction
                    goal_node = Node(
                        state=goal,
                        g_cost=goal_cost,
                        h_cost=0.0,
                        parent=new_node
                    )
                    self._best_path = self._reconstruct_path(goal_node)
                    print(f"    [RRT* Debug] Path found at iteration {iterations}, cost: {goal_cost:.2f}")
        
        elapsed_time = time.time() - start_time
        print(f"    [RRT* Debug] Planning completed in {elapsed_time:.2f}s after {iterations} iterations")
        
        if self._best_path:
            self._best_path = theta_smooth_path(self._best_path, self._obstacle_checker)
            # for i in range(len(self._best_path)):
            #     self._best_path[i] = (self._best_path[i][0], self._best_path[i][1], 0)
            print(f"    [RRT* Debug] Best path cost: {self._best_cost:.2f}")
            return self._best_path
        else:
            print(f"    [RRT* Debug] No path found")
            return None
    
    def _random_state(self) -> State2D:
        """Generate a random state within bounds."""
        x_min, x_max = self._world_bounds[0]
        y_min, y_max = self._world_bounds[1]
        
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        theta = random.uniform(-math.pi, math.pi)
        
        return np.array([x, y, theta])
    
    def _nearest_node(self, state: State2D) -> Node:
        """Find the nearest node in the tree to the given state."""
        min_distance = float('inf')
        nearest = self._tree_nodes[0]
        
        for node in self._tree_nodes:
            dist = self._distance(node.state, state)
            if dist < min_distance:
                min_distance = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_state: State2D, to_state: State2D) -> State2D:
        """Steer from one state towards another, limited by step size."""
        x1, y1, theta1 = from_state
        x2, y2, theta2 = to_state
        
        # Linear distance
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx**2 + dy**2)
        
        # Angle difference
        dtheta = self._angle_difference(theta2, theta1)
        
        # Limit motion by step sizes
        if dist > 0:
            step_ratio = min(1.0, self._step_size / dist)
            new_x = x1 + dx * step_ratio
            new_y = y1 + dy * step_ratio
        else:
            new_x = x1
            new_y = y1
        
        # Limit angle change
        new_theta: float = theta1
        if abs(dtheta) > 0:
            angle_ratio = min(1.0, self._angle_step_size / abs(dtheta))
            new_theta += dtheta * angle_ratio
        
        new_theta = self._normalize_angle(new_theta)
        
        return np.array([new_x, new_y, new_theta])
    
    def _distance(self, state1: State2D, state2: State2D) -> float:
        """Calculate Euclidean distance between two states (ignoring theta)."""
        x1, y1, _ = state1
        x2, y2, _ = state2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate smallest difference between two angles."""
        diff = angle1 - angle2
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _heuristic(self, state1: State2D, state2: State2D) -> float:
        """Heuristic: Euclidean distance in (x, y)."""
        return self._distance(state1, state2)
    
    def _calculate_rewire_radius(self, num_nodes: int) -> float:
        """Calculate rewiring radius based on number of nodes."""
        # Typical RRT* formula: r = eta * (log(n) / n)^(1/d)
        # where d is the dimension (3 in our case)
        if num_nodes < 2:
            return self._step_size * 2
        
        dimension: int = 3
        eta: float = self._rewire_radius_factor * self._step_size
        radius = eta * ((math.log(num_nodes) / num_nodes) ** (1 / dimension))
        
        return max(self._step_size, radius)
    
    def _find_nearby_nodes(self, state: State2D, radius: float) -> List[Node]:
        """Find all nodes within a given radius of the state."""
        nearby: List[Node] = []
        for node in self._tree_nodes:
            if self._distance(node.state, state) <= radius:
                nearby.append(node)
        return nearby
    
    def _reconstruct_path(self, node: Node) -> PathType:
        """Reconstruct path from start to given node."""
        path: PathType = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))