import numpy as np


from Types import PathType


class TrajectoryGenerator:
    """
    Converts a static spatial path (PathType) into a timed reference trajectory 
    for the MPC controller.
    """
    def __init__(self, dt: float, horizon: int, max_velocity: float = 2.0):
        self._dt: float = dt
        self._horizon: int = horizon
        self._max_velocity: float = max_velocity # m/s
        
    def get_reference_trajectory(self, current_state: np.ndarray, global_path: PathType) -> np.ndarray:
        """
        Generates a local reference trajectory (State_Size, Horizon) from the global path.
        
        Handles initial offset by optionally adding a reaching segment.
        Tracks angle continuously to avoid wrapping discontinuities.
        
        Args:
            current_state: The current robot state vector (x, y, theta, ...)
            global_path: List of (x, y, theta) waypoints from the planner
            
        Returns:
            ref_traj: (State_Size, Horizon) numpy array
        """
        #If the trajectory is empty, return a holding pattern at current position
        if not global_path or len(global_path) < 2:
            # Fallback: Stay in place if no path
            target: np.ndarray = current_state[:3] if len(current_state) >= 3 else np.zeros(3)
            '''a target state to hold'''
            state_dim: int = current_state.shape[0]
            ref: np.ndarray = np.zeros((state_dim, self._horizon))
            for i in range(3):
                ref[i, :] = target[i]
            return ref

        # Find closest point on path to robot
        path_np: np.ndarray = np.array(global_path)  # Shape (N, 3)
        path_xy: np.ndarray = path_np[:, :2]
        robot_xy: np.ndarray = current_state[:2]
        robot_theta: float = current_state[2]
        
        dists: np.ndarray = np.linalg.norm(path_xy - robot_xy, axis=1)
        closest_idx: int = np.argmin(dists) # type: ignore
        
        # Get closest point and next point on path
        p_closest: np.ndarray = np.array(global_path[closest_idx][:2])
        th_closest: float = global_path[closest_idx][2]
        
        # Handle initial offset: create reaching segment if robot is far from path
        ref_states: PathType = []
        initial_distance: float = np.linalg.norm(p_closest - robot_xy) # type: ignore
        
        if initial_distance > 0.5:  # If more than 0.5m away, add reaching segment
            num_reach_steps: int = min(5, self._horizon // 3)  # Use up to 5 steps or 1/3 of horizon
            for step in range(num_reach_steps):
                ratio: float = (step + 1) / num_reach_steps
                reach_x: float = robot_xy[0] + ratio * (p_closest[0] - robot_xy[0])
                reach_y: float = robot_xy[1] + ratio * (p_closest[1] - robot_xy[1])
                
                # Angle: smoothly transition to path angle
                reach_th_diff: float = th_closest - robot_theta
                while reach_th_diff > np.pi:
                    reach_th_diff -= 2 * np.pi
                while reach_th_diff < -np.pi:
                    reach_th_diff += 2 * np.pi
                reach_th: float = robot_theta + ratio * reach_th_diff
                
                ref_states.append([reach_x, reach_y, reach_th]) # type: ignore
        
        # Now continue with main path following
        current_path_idx: int = closest_idx
        step_dist: float = self._max_velocity * self._dt
        dist_accumulated: float = 0.0
        
        # Track continuous angle to avoid wrapping discontinuities
        continuous_theta: float = ref_states[-1][2] if ref_states else robot_theta
        
        for k in range(self._horizon - len(ref_states)):
            target_dist_from_start: float = (k + 1) * step_dist
            
            # Walk forward along path until we cover this distance
            while current_path_idx < len(global_path) - 1:
                p1: np.ndarray = np.array(global_path[current_path_idx][:2])
                p2: np.ndarray = np.array(global_path[current_path_idx + 1][:2])
                segment_len: float = np.linalg.norm(p2 - p1) # type: ignore
                
                if dist_accumulated + segment_len >= target_dist_from_start:
                    # Target point is on this segment
                    remaining: float = target_dist_from_start - dist_accumulated
                    ratio: float = remaining / segment_len if segment_len > 0 else 0
                    
                    # Interpolate position
                    interp_x: float = p1[0] + ratio * (p2[0] - p1[0])
                    interp_y: float = p1[1] + ratio * (p2[1] - p1[1])
                    
                    # Interpolate angle with continuous wrapping
                    th1: float = global_path[current_path_idx][2]
                    th2: float = global_path[current_path_idx + 1][2]
                    
                    # Find shortest angle difference
                    diff: float = th2 - th1
                    while diff > np.pi:
                        diff -= 2 * np.pi
                    while diff < -np.pi:
                        diff += 2 * np.pi
                    
                    # Interpolate
                    interp_th: float = th1 + ratio * diff
                    
                    # Make continuous: adjust to be closest to previous angle
                    if ref_states:
                        angle_diff: float = interp_th - continuous_theta
                        while angle_diff > np.pi:
                            angle_diff -= 2 * np.pi
                        while angle_diff < -np.pi:
                            angle_diff += 2 * np.pi
                        continuous_theta += angle_diff
                    else:
                        continuous_theta = interp_th
                    
                    ref_states.append([interp_x, interp_y, continuous_theta]) # type: ignore
                    break
                else:
                    # Segment too short, move to next
                    dist_accumulated += segment_len
                    current_path_idx += 1
            
            # If we ran out of path, repeat the goal
            if current_path_idx >= len(global_path) - 1:
                ref_states.append(global_path[-1])
        
        # Ensure we have exactly horizon steps
        while len(ref_states) < self._horizon:
            ref_states.append(global_path[-1])

        # Format for MPPI (State_Size, Horizon)
        ref_np: np.ndarray = np.array(ref_states).T  # Shape (3, Horizon)
        state_dim: int = current_state.shape[0]
        full_ref: np.ndarray = np.zeros((state_dim, self._horizon))
        
        full_ref[:3, :] = ref_np
        
        # Add velocity references: compute velocity from position differences
        for k in range(self._horizon):
            if k < self._horizon - 1:
                # Velocity = (next_pos - current_pos) / dt
                dx: float = ref_np[0, k+1] - ref_np[0, k]
                dy: float = ref_np[1, k+1] - ref_np[1, k]
                dtheta: float = ref_np[2, k+1] - ref_np[2, k]
                
                # Handle angle wrapping for velocity
                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi
                
                # Convert to robot frame velocity (approximate as global for now)
                vx: float = dx / self._dt
                vy: float = dy / self._dt
                omega: float = dtheta / self._dt
            else:
                # Last step: no further motion
                vx = vy = omega = 0.0
            
            # Set velocity references (indices 3, 4, 5)
            full_ref[3, k] = vx
            full_ref[4, k] = vy
            full_ref[5, k] = omega
        
        return full_ref