
from abc import ABC, abstractmethod
import math
import numpy as np
from typing import List, Tuple
from multiprocessing import Pool
from ObstacleDetection.Obstacle import Obstacle
from Scene.Robot import Robot
from Types import ConvexShape, State2D, Point2D


class ObstacleChecker(ABC):
    """Abstract base class for obstacle checking - implement with your obstacle detection."""
    
    @abstractmethod
    def is_collision(self, state: State2D) -> bool:
        """Check if the given state (x, y, theta) collides with obstacles."""
        pass
    
    @abstractmethod
    def is_path_clear(self, state1: State2D, state2: State2D) -> bool:
        """Check if the path between two states is collision-free."""
        pass

Robot_Geom = Tuple[List[Point2D], float, float, float, List[Tuple[float, float]]]
'''Tuple containing pre-computed robot geometry for collision checks:
- robot_corners: List of (x, y) robot corners in world frame
- robot_x, robot_y, robot_theta: Robot position and orientation
- axes: List of axes to test for SAT
'''

class StaticObstacleChecker(ObstacleChecker):
    """Concrete implementation of ObstacleChecker for static obstacles and rectangular robot."""
    def __init__(self, robot: Robot, obstacles: List[Obstacle], map_limits: Tuple[Tuple[float, float], Tuple[float, float]], use_parallelization: bool = False, collision_clearance: float = 0.0) -> None:
        super().__init__()
        self._robot: Robot = robot
        self._obstacles: List[Obstacle] = obstacles
        self._map_limits: Tuple[Tuple[float, float], Tuple[float, float]] = map_limits
         # ((x_min, x_max), (y_min, y_max))
        self._use_only_bounding_circles: bool = False
        '''If true, only use bounding circle checks for collision detection (faster, less accurate).'''
        self._use_parallelization: bool = use_parallelization and len(obstacles) > 5
        self._collision_clearance: float = collision_clearance
        '''Minimum clearance distance from obstacles (in meters). Robot must maintain this distance.'''
        
        # Pre-compute bounding circles for obstacles (for quick rejection)
        self._obstacle_bounds = self._compute_obstacle_bounds()
    
    def _compute_obstacle_bounds(self) -> List[Tuple[Point2D, float]]:
        """
        Pre-compute bounding circles for obstacles for quick rejection.
        Returns list of (center, radius) for each obstacle.
        """
        bounds = []
        for obs in self._obstacles:
            if obs.shape == ConvexShape.Circle and obs.center and obs.radius:
                # Circle: bounding circle is itself
                bounds.append((obs.center, obs.radius))
            elif obs.shape == ConvexShape.Polygon and obs.points:
                # Polygon: compute bounding circle
                points = np.array(obs.points)
                center = np.mean(points, axis=0)
                max_dist = np.max(np.linalg.norm(points - center, axis=1))
                bounds.append((tuple(center), max_dist))
            else:
                bounds.append((None, 0))
        return bounds
    
    def _precompute_robot_geometry(self, state: State2D) -> Robot_Geom:
        """
        Pre-compute robot geometry once per collision check to avoid redundant calculations.
        
        Returns:
            Tuple containing:
            - robot_corners: List of (x, y) robot corners in world frame
            - robot_x, robot_y, robot_theta: Robot position and orientation
            - collision_margin: Safety margin for collision detection
            - axes: List of axes to test for SAT
        """
        robot_x, robot_y, robot_theta = state
        
        half_length = self._robot.length / 2.0
        half_width = self._robot.width / 2.0
        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)
        
        # Get robot corners in local frame
        corners_local = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Transform to world frame
        robot_corners = []
        for lx, ly in corners_local:
            wx = robot_x + cos_theta * lx - sin_theta * ly
            wy = robot_y + sin_theta * lx + cos_theta * ly
            robot_corners.append((wx, wy))
        
        # Step 1: Get axes to test (normals of all edges)
        axes = []
        
        # Rectangle edge normals
        for i in range(4):
            p1 = robot_corners[i]
            p2 = robot_corners[(i + 1) % 4]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Perpendicular (normal) to edge
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-6:
                axes.append((normal[0] / length, normal[1] / length))
        
        return robot_corners, robot_x, robot_y, robot_theta, axes
        
    def _point_to_segment_distance(self, point: Point2D, seg_start: Point2D, seg_end: Point2D) -> float:
        """
        Calculate minimum distance from a point to a line segment.
        This is the core of the fast wall collision detection.
        
        Math:
            - Project point onto infinite line defined by segment
            - Clamp projection parameter t to [0, 1] to stay on segment
            - Return distance to clamped point
        
        Args:
            point: (x, y) point to measure from
            seg_start: (x, y) segment start
            seg_end: (x, y) segment end
            
        Returns:
            Minimum distance from point to segment
        """
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        # Vector from segment start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from segment start to point
        px_dx = px - x1
        py_dy = py - y1
        
        # If segment is degenerate (start == end), return distance to that point
        denom = dx * dx + dy * dy
        if denom < 1e-10:
            return math.sqrt(px_dx**2 + py_dy**2)
        
        # Parameter t of closest point on infinite line
        # t = 0 at seg_start, t = 1 at seg_end
        t = (px_dx * dx + py_dy * dy) / denom
        
        # Clamp t to [0, 1] to stay on segment
        t = max(0.0, min(1.0, t))
        
        # Find closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point on segment
        dist_x = px - closest_x
        dist_y = py - closest_y
        return math.sqrt(dist_x**2 + dist_y**2)
    
    def _is_out_of_bounds(self, state: State2D) -> bool:
        """
        Check if any part of the robot is outside the map bounds.
        
        Args:
            state: Robot state (x, y, theta)
            
        Returns:
            True if robot is out of bounds, False otherwise
        """
        if self._map_limits is None:
            return False
        
        (x_min, x_max), (y_min, y_max) = self._map_limits
        
        # Pre-compute robot geometry to get all corners
        robot_corners, robot_x, robot_y, robot_theta, _ = self._precompute_robot_geometry(state)
        
        # Check if any corner is outside bounds
        for corner_x, corner_y in robot_corners:
            if corner_x < x_min or corner_x > x_max or corner_y < y_min or corner_y > y_max:
                return True
        
        return False
    
    def _quick_rejection_check(self, state: State2D, obstacle_idx: int) -> bool:
        """
        Quick bounding circle check before expensive collision detection.
        Returns True if definitely NO collision, False if might collide (needs full check).
        """
        robot_x, robot_y, theta = state
        center, radius = self._obstacle_bounds[obstacle_idx]
        
        if center is None:
            return False
        
        # Robot bounding circle (approximation: diagonal/2)
        robot_radius = math.sqrt(self._robot.length**2 + self._robot.width**2) / 2
        
        # Distance between centers
        dist = math.sqrt((robot_x - center[0])**2 + (robot_y - center[1])**2)
        
        # If too far apart, no collision possible (accounting for clearance)
        return dist > (robot_radius + radius + self._collision_clearance + 1.0)  # +1.0 safety margin
    
    def _check_obstacle_collision(self, obstacle_idx: int, state: State2D, robot_geom: Robot_Geom) -> bool:
        """
        Check collision with a single obstacle using pre-computed robot geometry.
        
        Args:
            obstacle_idx: Index of obstacle to check
            state: Robot state (x, y, theta)
            robot_geom: Pre-computed robot geometry tuple from _precompute_robot_geometry()
        """
        obs = self._obstacles[obstacle_idx]
        
        # Quick rejection: if bounding circles don't overlap, no collision possible
        if self._quick_rejection_check(state, obstacle_idx):
            return False
        
        # If only using bounding circles for collision check, assume collision if we got here, otherwise would have rejected in quick check
        if self._use_only_bounding_circles:
            return True
        
        if obs.shape == ConvexShape.Circle:
            if self._circle_rectangle_collision(
                circle_center=obs.center,  # type: ignore
                circle_radius=obs.radius,  # type: ignore
                robot_state=state
            ):
                return True
                
        elif obs.shape == ConvexShape.Polygon:
            if not obs.points or len(obs.points) < 3:
                return False
            
            if self._polygon_rectangle_collision(
                polygon_points=obs.points,
                robot_geom=robot_geom
            ):
                return True
        
        return False
    
    def _polygon_rectangle_collision(self, polygon_points: List[Point2D], robot_geom: Robot_Geom) -> bool:
        """
        Check collision between rectangular robot and single polygon obstacle using Separating Axis Theorem (SAT).
        Accounts for collision clearance by expanding the rectangle.
        
        Args:
            polygon_points: List[Point2D], 
            robot_geom: Pre-computed robot geometry tuple from _precompute_robot_geometry()
        Returns:
            True if collision detected (including clearance violation), False otherwise
        """
        rect_corners_world, robot_x, robot_y, robot_theta, axes = robot_geom
        
        # Expand rectangle corners by clearance distance from center
        if self._collision_clearance > 0:
            expanded_corners = []
            for corner in rect_corners_world:
                # Vector from robot center to corner
                to_corner_x = corner[0] - robot_x
                to_corner_y = corner[1] - robot_y
                
                # Distance from center to corner
                dist = math.sqrt(to_corner_x**2 + to_corner_y**2)
                
                if dist > 1e-6:
                    # Normalize and expand corner outward
                    dir_x = to_corner_x / dist
                    dir_y = to_corner_y / dist
                    expanded_corner = (
                        corner[0] + dir_x * self._collision_clearance,
                        corner[1] + dir_y * self._collision_clearance
                    )
                else:
                    expanded_corner = corner
                    
                expanded_corners.append(expanded_corner)
            rect_corners_world = expanded_corners
        
        # Polygon edge normals
        for i in range(len(polygon_points)):
            p1 = polygon_points[i]
            p2 = polygon_points[(i + 1) % len(polygon_points)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-6:
                axes.append((normal[0] / length, normal[1] / length))
        
        # Step 2: Project both shapes onto each axis
        for axis in axes:
            # Project rectangle
            rect_projections = []
            for corner in rect_corners_world:
                proj = corner[0] * axis[0] + corner[1] * axis[1]
                rect_projections.append(proj)
            rect_min = min(rect_projections)
            rect_max = max(rect_projections)
            
            # Project polygon
            poly_projections = []
            for point in polygon_points:
                proj = point[0] * axis[0] + point[1] * axis[1]
                poly_projections.append(proj)
            poly_min = min(poly_projections)
            poly_max = max(poly_projections)
            
            # Check for gap (separating axis found)
            if rect_max < poly_min or poly_max < rect_min:
                return False  # No collision
        
        # No separating axis found
        return True
    
    def _circle_rectangle_collision(self, circle_center: Point2D, circle_radius: float, robot_state: State2D) -> bool:
        """
        Check collision between a rotated rectangle (robot) and a circle (obstacle).
        Accounts for collision clearance distance.
        
        Args:
            circle_center: (x, y) position of circle
            circle_radius: radius of circle
            robot_state: (x, y, theta) position and orientation of robot
            
        Returns:
            True if collision detected (including clearance violation), False otherwise
        """
        robot_x, robot_y, robot_theta = robot_state
        circle_x, circle_y = circle_center
        
        # Step 1: Transform circle center to robot's local frame
        # Translate circle relative to robot center
        dx = circle_x - robot_x
        dy = circle_y - robot_y
        
        # Rotate into robot's frame (inverse rotation)
        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)
        
        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy
        
        # Step 2: Find closest point on rectangle to circle center (in robot's frame)
        # Rectangle extends from -length/2 to +length/2 in x, -width/2 to +width/2 in y
        half_length = self._robot.length / 2.0
        half_width = self._robot.width / 2.0
        
        # Clamp circle center to rectangle bounds
        closest_x = max(-half_length, min(local_x, half_length))
        closest_y = max(-half_width, min(local_y, half_width))
        
        # Step 3: Calculate distance from circle center to closest point
        dist_x = local_x - closest_x
        dist_y = local_y - closest_y
        distance = math.sqrt(dist_x**2 + dist_y**2)
        
        # Step 4: Check collision (including clearance)
        return distance < (circle_radius + self._collision_clearance)
        
    def is_collision(self, state: State2D) -> bool:
        """
        Check if robot at given state collides with any obstacle or is out of bounds.
        Pre-computes robot geometry once, then checks all obstacles.
        
        Args:
            state: Robot state (x, y, theta)
            
        Returns:
            True if collision detected or out of bounds, False otherwise
        """
        # Check bounds first (fast early exit)
        if self._is_out_of_bounds(state):
            return True
        
        # Pre-compute robot geometry once to avoid recalculating for each obstacle
        robot_geom = self._precompute_robot_geometry(state)
        
        if self._use_parallelization and len(self._obstacles) >= 10:
            # Parallel check using multiprocessing
            with Pool(processes=4) as pool:
                results = pool.starmap(
                    self._check_obstacle_collision,
                    [(i, state, robot_geom) for i in range(len(self._obstacles))]
                )
                return any(results)
        else:
            # Sequential check with early exit (faster for small obstacle count)
            for i in range(len(self._obstacles)):
                if self._check_obstacle_collision(i, state, robot_geom):
                    return True
            return False
    
    # TODO: use num_samples from parameters
    def is_path_clear(self, state1: State2D, state2: State2D, num_samples: int = 20) -> bool:
        """
        Check if a straight-line path between two states is collision-free.
        
        Uses linear interpolation between states and samples at multiple points
        along the path to detect collisions.
        
        Args:
            state1: Starting state (x1, y1, theta1)
            state2: Ending state (x2, y2, theta2)
            num_samples: Number of points to sample along the path (higher = more accurate)
            
        Returns:
            True if path is clear, False if collision detected
        """
        x1, y1, theta1 = state1
        x2, y2, theta2 = state2
        
        # Sample points along the path
        for i in range(num_samples + 1):
            t = i / num_samples  # Parameter from 0 to 1
            
            # Linear interpolation
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Interpolate angle (handle wrapping)
            angle_diff = theta2 - theta1
            # Normalize angle difference to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            theta = theta1 + t * angle_diff
            
            sample_state = (x, y, theta)
            
            # Check collision at this sample point
            if self.is_collision(sample_state):
                return False  # Path blocked
        
        return True  # Path is clear
    
    def _get_single_obstacle_distance(self, obstacle_idx: int, state: State2D, robot_geom: Robot_Geom) -> float:
        """
        Compute minimum distance from robot to a single obstacle.
        Returns the actual distance minus the clearance requirement.
        
        Args:
            obstacle_idx: Index of obstacle
            state: Robot state (x, y, theta)
            robot_geom: Pre-computed robot geometry tuple
            
        Returns:
            Minimum distance to obstacle minus clearance (negative if violation)
        """
        obs = self._obstacles[obstacle_idx]
        robot_x, robot_y, robot_theta = state
        robot_corners, _, _, _, _ = robot_geom
        
        min_dist: float = float('inf')
        
        if obs.shape == ConvexShape.Circle:
            # Distance from robot center to circle center minus radii
            center_dist = math.sqrt((robot_x - obs.center[0])**2 + (robot_y - obs.center[1])**2)  # type: ignore
            robot_radius = math.sqrt(self._robot.length**2 + self._robot.width**2) / 2
            min_dist = center_dist - robot_radius - obs.radius  # type: ignore
                
        elif obs.shape == ConvexShape.Polygon and obs.points:
            if len(obs.points) < 3:
                return float('inf')
            
            # Calculate min distance from robot corners to polygon edges, and vice versa
            # Robot corners to polygon edges
            for corner in robot_corners:
                for i in range(len(obs.points)):
                    p1 = obs.points[i]
                    p2 = obs.points[(i + 1) % len(obs.points)]
                    dist = self._point_to_segment_distance(corner, p1, p2)
                    min_dist = min(min_dist, dist)
            
            # Polygon vertices to robot edges
            for vertex in obs.points:
                for i in range(4):
                    p1 = robot_corners[i]
                    p2 = robot_corners[(i + 1) % 4]
                    dist = self._point_to_segment_distance(vertex, p1, p2)
                    min_dist = min(min_dist, dist)
        
        # Subtract clearance from distance to get remaining safety margin
        return min_dist - self._collision_clearance
    
    def get_obstacle_distances(self, state: State2D) -> List[float]:
        """
        Get minimum distance from robot to each obstacle.
        
        Reuses the same collision detection geometry and algorithms. Can use fast
        bounding circle approximation if use_only_bounding_circles is True.
        
        Args:
            state: Robot state (x, y, theta)
            
        Returns:
            List of minimum distances to each obstacle (same order as self._obstacles)
            Negative distance means collision (penetration depth estimate)
        """
        robot_x, robot_y, robot_theta = state
        
        # Fast path: use only bounding circles (very fast, less accurate)
        if self._use_only_bounding_circles:
            distances = []
            for obs in self._obstacles:
                center, radius = self._obstacle_bounds[self._obstacles.index(obs)]
                if center is None:
                    distances.append(float('inf'))
                else:
                    # Distance between centers minus radii
                    robot_radius = math.sqrt(self._robot.length**2 + self._robot.width**2) / 2
                    center_dist = math.sqrt((robot_x - center[0])**2 + (robot_y - center[1])**2)
                    distances.append(center_dist - robot_radius - radius)
            return distances
        
        # Accurate path: use pre-computed geometry and precise collision checks
        robot_geom: Robot_Geom = self._precompute_robot_geometry(state)
        
        if self._use_parallelization and len(self._obstacles) >= 10:
            # Parallel computation using multiprocessing
            with Pool(processes=4) as pool:
                distances = pool.starmap(self._get_single_obstacle_distance, [(i, state, robot_geom) for i in range(len(self._obstacles))])
                return distances
        else:
            # Sequential computation with early exit capability
            distances = []
            
            for i in range(len(self._obstacles)):
                dist = self._get_single_obstacle_distance(i, state, robot_geom)
                distances.append(dist)
            
            return distances
