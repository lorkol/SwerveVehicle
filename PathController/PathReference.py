import numpy as np
from Types import Point2D

from ObstacleDetection.ObstacleDetector import ObstacleChecker

class ProjectedPathFollower:
    def __init__(self, path_points: list, obstacle_checker: ObstacleChecker = None, min_obstacle_dist: float = 0.5, max_obstacle_dist: float = 3.0, min_velocity_scale: float = 0.2):
        """
        Initialize the path follower.
        
        Args:
            path_points: List of points [x, y] or [x, y, theta]
            obstacle_checker: Optional obstacle checker for distance-based velocity scaling
            min_obstacle_dist: Distance at which velocity is scaled to minimum (meters)
            max_obstacle_dist: Distance beyond which velocity is not scaled (meters)
            min_velocity_scale: Minimum velocity scale factor (0.0 to 1.0) when at min_obstacle_dist
        """
        # Convert to numpy array for fast math
        # Expects points as [x, y] or [x, y, theta]
        self.path = np.array(path_points)
        self.last_idx: int = 0
        
        # Obstacle-aware velocity scaling
        self.obstacle_checker = obstacle_checker
        self.min_obstacle_dist = min_obstacle_dist
        self.max_obstacle_dist = max_obstacle_dist
        self.min_velocity_scale = min_velocity_scale
    
    def _get_velocity_scale(self, current_pos: np.ndarray) -> float:
        """
        Compute velocity scale factor based on distance to closest obstacle.
        
        Returns:
            Scale factor between min_velocity_scale and 1.0
            - Returns 1.0 if no obstacle checker or obstacles are far away
            - Returns min_velocity_scale if closest obstacle is at min_obstacle_dist or closer
            - Linearly interpolates between for intermediate distances
        """
        if self.obstacle_checker is None:
            return 1.0
                
        state = (float(current_pos[0]), float(current_pos[1]), float(current_pos[2]))
        
        # Get distances to all obstacles
        distances = self.obstacle_checker.get_obstacle_distances(state)
        
        if not distances:
            return 1.0
        
        # Find minimum distance to any obstacle
        min_distance = min(distances)
        
        # Clamp distance to valid range
        if min_distance >= self.max_obstacle_dist:
            return 1.0
        elif min_distance <= self.min_obstacle_dist:
            return self.min_velocity_scale
        else:
            # Linear interpolation between min and max velocity scale
            t = (min_distance - self.min_obstacle_dist) / (self.max_obstacle_dist - self.min_obstacle_dist)
            return self.min_velocity_scale + t * (1.0 - self.min_velocity_scale)
        
    def get_reference_state(self, current_pos: np.ndarray, lookahead_dist: float, v_desired: float) -> np.ndarray:
        """
        Returns the reference state vector [x_ref, y_ref, th_ref, vx_ref, vy_ref, vth_ref]
        based on projecting current_pos onto the path and looking ahead.
        """
        
        # 1. Find Closest Point on Path Segment
        # Key: progress is MONOTONIC - only search forward from last_idx
        # This ensures we always move toward the goal, never backward
        # Use larger search window to ensure we can reach the goal
        remaining_segments = len(self.path) - 1 - self.last_idx
        search_window = max(10, remaining_segments)  # Search all remaining segments if needed
        start_search = self.last_idx  # Only search forward
        end_search = min(self.last_idx + search_window, len(self.path) - 1)
        
        min_dist = float('inf')
        best_point = self.path[self.last_idx][:2]
        best_idx = self.last_idx

        for i in range(start_search, end_search):
            p1 = self.path[i]
            p2 = self.path[i+1]
            
            proj, t = self._project_on_segment(current_pos, p1, p2)
            dist = np.linalg.norm(proj - current_pos[:2])
            
            if dist < min_dist:
                min_dist = dist
                best_point = proj
                best_idx = i
        
        # Advance logic: if we've passed the current segment (t > 0.9), move to next
        # This ensures monotonic progress even when off-path
        if best_idx < len(self.path) - 2:
            _, t = self._project_on_segment(current_pos, self.path[best_idx], self.path[best_idx + 1])
            if t > 0.9:  # Passed 90% of segment
                best_idx = min(best_idx + 1, len(self.path) - 2)
                best_point, _ = self._project_on_segment(current_pos, self.path[best_idx], self.path[best_idx + 1])

        # Ensure we never go backward
        self.last_idx = max(self.last_idx, best_idx)
        best_idx = self.last_idx
        
        # Re-project onto current segment
        best_point, _ = self._project_on_segment(current_pos, self.path[best_idx], self.path[best_idx + 1])
        
        # Check if we're at the end of the path
        goal_point = self.path[-1][:2]
        goal_theta = self.path[-1][2] if len(self.path[-1]) > 2 else 0.0
        dist_to_goal = np.linalg.norm(current_pos[:2] - goal_point)
        
        # Goal mode: when on last segment, always target the goal
        # This ensures the robot reaches the goal even if it overshoots
        on_last_segment = best_idx >= len(self.path) - 2
        
        if on_last_segment:
            # Provide velocity reference toward goal, scaled by distance
            # Velocity naturally approaches zero as robot approaches goal
            if dist_to_goal > 1e-6:
                dir_to_goal = goal_point - current_pos[:2]
                dir_normalized = dir_to_goal / dist_to_goal
                
                velocity_scale = self._get_velocity_scale(current_pos)
                # Velocity proportional to distance - naturally ramps down
                approach_velocity = min(v_desired * velocity_scale, dist_to_goal)
                
                ref_vel = dir_normalized * approach_velocity
                return np.array([goal_point[0], goal_point[1], goal_theta, ref_vel[0], ref_vel[1], 0.0])
            else:
                return np.array([goal_point[0], goal_point[1], goal_theta, 0.0, 0.0, 0.0])
        
        # Get Current Segment Tangent
        p1 = self.path[best_idx][:2]
        p2_full = self.path[best_idx+1] if best_idx + 1 < len(self.path) else self.path[best_idx]
        p2 = p2_full[:2]
        
        vec1 = p2 - p1
        len1 = np.linalg.norm(vec1)
        if len1 > 1e-6:
            tangent1 = vec1 / len1
        else:
            tangent1 = np.array([1.0, 0.0])

        # Get Next Segment Tangent (Lookahead for curvature)
        # We look at the NEXT segment to anticipate the turn
        next_idx = min(best_idx + 1, len(self.path) - 2)
        p3_full = self.path[next_idx + 1]
        p3 = p3_full[:2]
        
        vec2 = p3 - p2
        len2 = np.linalg.norm(vec2)
        if len2 > 1e-6:
            tangent2 = vec2 / len2
        else:
            tangent2 = tangent1 # No turn

        # Calculate Change in Angle (d_theta)
        # Use cross product and dot product to find signed angle difference
        # angle = atan2(sin_angle, cos_angle)
        # cross_product_2d = x1*y2 - y1*x2
        cross_prod = tangent1[0] * tangent2[1] - tangent1[1] * tangent2[0]
        dot_prod = np.dot(tangent1, tangent2)
        d_theta = np.arctan2(cross_prod, dot_prod)

        # Calculate Curvature (kappa = d_theta / distance)
        # We assume this turn happens over the length of the current segment
        # (Or you can smooth it over 'lookahead_dist')
        if len1 > 1e-6:
            kappa = d_theta / len1
        else:
            kappa = 0.0

        # Apply obstacle-aware velocity scaling
        velocity_scale = self._get_velocity_scale(current_pos)
        scaled_v_desired = v_desired * velocity_scale

        # Calculate Reference Angular Velocity
        # v_theta = curvature * linear_velocity
        ref_v_theta = kappa * scaled_v_desired

        # --- Standard Position/Velocity Calculation ---
        # Compute lookahead reference but clamp it so it does not go past
        # the end of the final segment (prevents reference being beyond goal).
        if best_idx + 1 >= len(self.path) - 1:
            # We're on the final segment - ensure we don't overshoot the goal
            goal_pt = self.path[-1][:2]
            remaining = np.dot(goal_pt - best_point, tangent1)
            remaining = max(0.0, remaining)
            lookahead_use = min(lookahead_dist, remaining)
            ref_pos = best_point + tangent1 * lookahead_use
        else:
            ref_pos = best_point + tangent1 * lookahead_dist

        ref_theta = np.arctan2(tangent1[1], tangent1[0])
        ref_vel = tangent1 * scaled_v_desired

        # Return updated state with v_theta
        return np.array([ref_pos[0], ref_pos[1], ref_theta, ref_vel[0], ref_vel[1], ref_v_theta ])
        
    def _project_on_segment(self, point, p1, p2):
        # Extract x, y coordinates only (ignore theta)
        point_2d = point[:2]
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        
        seg_v = p2_2d - p1_2d
        pt_v = point_2d - p1_2d
        seg_len_sq = np.dot(seg_v, seg_v)
        
        if seg_len_sq == 0:
            return p1_2d, 0.0
            
        t = np.dot(pt_v, seg_v) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        return p1_2d + t * seg_v, t