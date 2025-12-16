
from ActuatorController.ActuatorController import ActuatorController
from PathController.LocalPlanners.LocalPlanner import LocalPlanner
from Types import State2D, State6D, NP3DPoint
from typing import List, Tuple, Optional
import numpy as np
import math


class PurePursuitController(LocalPlanner):
    def __init__(self, robot_controller: ActuatorController, path_points: List[State2D], lookahead: float = 0.5, v_desired: float = 1.0, dt: float = 0.1):
        """
        Pure Pursuit Reference Generator for Cascaded Control.
        
        Args:
            path_points: List of [x, y, theta] points (Theta from planner for obstacle avoidance)
            lookahead_dist: Radius of the search circle (The "Carrot" distance)
            v_desired: Cruise velocity for the robot
        """
        super().__init__(robot_controller=robot_controller, path_points=path_points)
        
        self._lookahead: float = lookahead  # meters
        self._v_desired: float = v_desired  # m/s
        self._dt: float = dt  # seconds
            # State tracking to prevent searching backwards
        self._last_index: int = 0 
        self._max_search_window: int = 100 # Optimize search for long paths # TODO: Get from parameters                  
        
    def get_reference_state(self, current_pose: NP3DPoint, debug: bool = False) -> State6D:
        """
        Calculates the 6D Reference State for the LQR Controller.
        
        Args:
            current_pose: Robot state [x, y, theta]
            
        Returns:
            ref_state: [x_carrot, y_carrot, theta_path, v_x, v_y, omega_ref]
            Note: The Next step in the Cascading Controller will ignore x_carrot/y_carrot because you mask them,
            but we return them for visualization/debug.
        """
        rx, ry = current_pose[0], current_pose[1]

        # --- 1. Find the Lookahead Point (Carrot) ---
        carrot_idx, carrot_point = self._find_lookahead_point(rx, ry, self._lookahead, debug)
        if debug: print(f"[PurePursuit] Current pose: {current_pose}, Carrot idx: {carrot_idx}, Carrot point: {carrot_point}")
        
        # Update index so we don't search behind us next time
        self._last_index = carrot_idx

        # --- 2. Calculate Velocity Vector (Translation) ---
        # Vector from Robot -> Carrot
        dx = carrot_point[0] - rx
        dy = carrot_point[1] - ry
        dist = np.hypot(dx, dy)
        if debug: print(f"[PurePursuit] dx: {dx}, dy: {dy}, dist: {dist}")

        if dist > 1e-3:
            # Scale vector to desired speed
            # This is the "Slide" vector
            vx_ref = (dx / dist) * self._v_desired
            vy_ref = (dy / dist) * self._v_desired
        else:
            # We are ON the carrot (end of path)
            vx_ref, vy_ref = 0.0, 0.0

        # --- 3. Retrieve Heading (Rotation) ---
        # We take theta directly from the path planner (Point index 2)
        # This allows independent rotation for obstacle avoidance

        # --- 4. Construct Reference State ---
        # [x_ref, y_ref, theta_ref, vx_ref, vy_ref, vtheta_ref=0.0]
        ref_state = np.array([carrot_point[0], carrot_point[1], carrot_point[2], vx_ref, vy_ref, 0.0])
        if debug: print(f"[PurePursuit] Reference state: {ref_state}")
        return ref_state

    def _find_lookahead_point(self, rx: float, ry: float, L: float, debug: bool = False) -> Tuple[int, np.ndarray]:
        """
        Finds the furthest point on the path that intersects the lookahead circle.
        """
        # Limit search window to maintain efficiency
        end_search: int = min(self._last_index + self._max_search_window, len(self.path))
        
        best_idx: int = self._last_index
        best_point: NP3DPoint = np.array(self.path[self._last_index])
        found_intersection: bool = False
        final_point = self.path[-1]
        dist_to_end = np.hypot(final_point[0] - rx, final_point[1] - ry)
        
        if dist_to_end < L:
            return len(self.path) - 1, final_point

        # Iterate through path segments
        for i in range(self._last_index, end_search - 1):
            p1: NP3DPoint = np.array(self.path[i])     # Start of segment
            p2: NP3DPoint = np.array(self.path[i + 1]) # End of segment
            
            # Check intersection between Segment (p1, p2) and Circle (rx, ry, L)
            intersections: List[NP3DPoint] = self._circle_segment_intersection(p1, p2, rx, ry, L)
            
            if intersections:
                # If valid intersections exist, pick the one furthest along the segment
                # (The one closest to p2)
                found_intersection = True
                best_idx = i
                best_point = intersections[-1] # List is sorted by distance from p1
                if debug: print(f"[PurePursuit] Intersection found at segment {i}: {intersections}, best_idx: {best_idx}, best_point: {best_point}")
        # Fallback: If no intersection found (too far or too close)
        if not found_intersection:
            # Check distance to the last point we found
            dist_to_current: float = np.hypot(self.path[self._last_index][0] - rx, self.path[self._last_index][1] - ry)
            if debug: print(f"[PurePursuit] No intersection found. dist_to_current: {dist_to_current}, L: {L}")
            if dist_to_current > L and debug:
                # We are too far away -> Cut inward to the nearest point (Survival mode)
                # Or just keep the current target (Wait for it)
                print(f"[PurePursuit] Too far from path, keeping current target at index {self._last_index}")
            else:
                # We are closer than L -> Look forward for the first point outside L
                for i in range(self._last_index, end_search):
                    p: NP3DPoint = np.array(self.path[i])
                    d: float = np.hypot(p[0] - rx, p[1] - ry)
                    if d > L:
                        best_idx = i
                        best_point = p
                        if debug: print(f"[PurePursuit] Found first point outside L at index {i}: {p}")
                        break
        if debug: print(f"[PurePursuit] Returning best_idx: {best_idx}, best_point: {best_point}")
        return best_idx, best_point

    def _circle_segment_intersection(self, p1: NP3DPoint, p2: NP3DPoint, rx: float, ry: float, r: float) -> List[NP3DPoint]:
        """
        Math for finding intersection between a line segment and a circle.
        Returns a list of intersection points [x, y, theta] (interpolating theta).
        """
        # Shift coordinates so circle is at (0,0)
        x1, y1 = p1[0] - rx, p1[1] - ry
        x2, y2 = p2[0] - rx, p2[1] - ry
        
        dx: float = x2 - x1
        dy: float = y2 - y1
        dr: float = math.sqrt(dx**2 + dy**2)
        
        # NOTE: If p1, p2 share x, y coords and differ only in theta
        eps = 1e-8
        if dr < eps:
            # p1 and p2 are at same (x,y) in circle-centered coords
            dist_point = math.hypot(x1, y1)  # distance from circle center to the point
            if abs(dist_point - r) <= 1e-6:
                # point lies on circle: return intersection at world coords
                cx_world = x1 + rx
                cy_world = y1 + ry
                # choose p2 theta (or return both [p1[2], p2[2]] if desired)
                return [np.array([cx_world, cy_world, p2[2]])]
            return []  # no intersection
        
        D: float = x1 * y2 - x2 * y1
        
        discriminant: float = r**2 * dr**2 - D**2
        
        if discriminant < 0:
            return [] # No intersection
            
        sqrt_disc: float = math.sqrt(discriminant)
        
        # Calculate two possible intersection points (infinite line)
        sol_x1: float = (D * dy + (1 if dy >= 0 else -1) * dx * sqrt_disc) / dr**2
        sol_x2: float = (D * dy - (1 if dy >= 0 else -1) * dx * sqrt_disc) / dr**2
        
        sol_y1: float = (-D * dx + abs(dy) * sqrt_disc) / dr**2
        sol_y2: float = (-D * dx - abs(dy) * sqrt_disc) / dr**2
        
        candidates: List[Tuple[float, float]] = [(sol_x1 + rx, sol_y1 + ry), (sol_x2 + rx, sol_y2 + ry)]
        valid_intersections: List[NP3DPoint] = []
        
        # Check which points are actually on the segment p1-p2
        # We can use a dot product or bounding box check
        min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
        min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
        epsilon: float = 1e-5
        
        for (cx, cy) in candidates:
            if (min_x - epsilon <= cx <= max_x + epsilon) and \
               (min_y - epsilon <= cy <= max_y + epsilon):
                   
                # Interpolate Theta based on distance
                dist_p1: float = np.hypot(cx - p1[0], cy - p1[1])
                total_dist: float = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                ratio: float = dist_p1 / total_dist if total_dist > 0 else 0
                
                # Interpolate angle (handle wrapping)
                th1, th2 = p1[2], p2[2]
                diff: float = (th2 - th1 + np.pi) % (2*np.pi) - np.pi
                c_theta: float = th1 + diff * ratio
                
                valid_intersections.append(np.array([cx, cy, c_theta]))
                
        # Sort by distance from p1 so we pick the furthest one along the segment
        valid_intersections.sort(key=lambda p: np.hypot(p[0] - p1[0], p[1] - p1[1]))
        
        return valid_intersections