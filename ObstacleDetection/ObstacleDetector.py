
from abc import ABC, abstractmethod
import math
import numpy as np
from typing import List
from ObstacleDetection.Obstacle import Obstacle
from Scene.Robot import Robot
from Types import ConvexShape, State, Point2D


class ObstacleChecker(ABC):
    """Abstract base class for obstacle checking - implement with your obstacle detection."""
    
    @abstractmethod
    def is_collision(self, state: State) -> bool:
        """Check if the given state (x, y, theta) collides with obstacles."""
        pass
    
    @abstractmethod
    def is_path_clear(self, state1: State, state2: State) -> bool:
        """Check if the path between two states is collision-free."""
        pass


class StaticObstacleChecker(ObstacleChecker):
    """Concrete implementation of ObstacleChecker for static obstacles and rectangular robot."""
    def __init__(self, robot: Robot, obstacles: List[Obstacle]) -> None:
        super().__init__()
        self._robot: Robot = robot
        self._obstacles: List[Obstacle] = obstacles
    
    def _polygon_rectangle_collision(self, polygon_points: List[Point2D], robot_state: State, robot_length: float, robot_width: float) -> bool:
        """
        Check collision between a rotated rectangle (robot) and a polygon (obstacle)
        using Separating Axis Theorem (SAT).
        
        Args:
            polygon_points: List of (x, y) vertices of the polygon
            robot_state: (x, y, theta) position and orientation of robot
            robot_length: length of robot
            robot_width: width of robot
            
        Returns:
            True if collision detected, False otherwise
        """
        robot_x, robot_y, robot_theta = robot_state
        
        # Get rectangle corners in world frame
        half_length = robot_length / 2.0
        half_width = robot_width / 2.0
        
        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)
        
        # Rectangle corners in local frame
        rect_corners_local = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Transform rectangle corners to world frame
        rect_corners_world = []
        for lx, ly in rect_corners_local:
            wx = robot_x + cos_theta * lx - sin_theta * ly
            wy = robot_y + sin_theta * lx + cos_theta * ly
            rect_corners_world.append((wx, wy))
        
        # Step 1: Get axes to test (normals of all edges)
        axes = []
        
        # Rectangle edge normals
        for i in range(4):
            p1 = rect_corners_world[i]
            p2 = rect_corners_world[(i + 1) % 4]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Perpendicular (normal) to edge
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-6:
                axes.append((normal[0] / length, normal[1] / length))
        
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
    
    def _circle_rectangle_collision(self, circle_center: Point2D, circle_radius: float, robot_state: State, robot_length: float, robot_width: float) -> bool:
        """
        Check collision between a rotated rectangle (robot) and a circle (obstacle).
        
        Args:
            circle_center: (x, y) position of circle
            circle_radius: radius of circle
            robot_state: (x, y, theta) position and orientation of robot
            robot_length: length of robot (along x-axis when theta=0)
            robot_width: width of robot (along y-axis when theta=0)
            
        Returns:
            True if collision detected, False otherwise
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
        half_length = robot_length / 2.0
        half_width = robot_width / 2.0
        
        # Clamp circle center to rectangle bounds
        closest_x = max(-half_length, min(local_x, half_length))
        closest_y = max(-half_width, min(local_y, half_width))
        
        # Step 3: Calculate distance from circle center to closest point
        dist_x = local_x - closest_x
        dist_y = local_y - closest_y
        distance = math.sqrt(dist_x**2 + dist_y**2)
        
        # Step 4: Check collision
        return distance < circle_radius
        
    def is_collision(self, state: State) -> bool:
        """
        Check if robot at given state collides with any obstacle.
        
        Args:
            state: Robot state (x, y, theta)
            
        Returns:
            True if collision detected, False otherwise
        """
        for obs in self._obstacles:
            if obs.shape == ConvexShape.Circle:
                # Circle obstacle              
                if self._circle_rectangle_collision(
                    circle_center=obs.center, # type: ignore
                    circle_radius=obs.radius, # type: ignore
                    robot_state=state,
                    robot_length=self._robot.length,
                    robot_width=self._robot.width
                ):
                    return True
                    
            elif obs.shape == ConvexShape.Polygon:
                # Polygon obstacle using Separating Axis Theorem
                if not obs.points or len(obs.points) < 3:
                    continue
                
                if self._polygon_rectangle_collision(
                    polygon_points=obs.points,
                    robot_state=state,
                    robot_length=self._robot.length,
                    robot_width=self._robot.width
                ):
                    return True
                
        return False
    
    def is_path_clear(self, state1: State, state2: State, num_samples: int = 20) -> bool:
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