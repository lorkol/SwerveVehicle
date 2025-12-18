import pygame
import numpy as np
import math
import matplotlib.pyplot as plt

# Import your existing Simulation class
from Testing import Simulation
from Uncertainties.uncertainty import add_force_uncertainty, add_state_estimation_uncertainty
from Types import ConvexShape
from PathController.Robot_Sim import Robot_Sim

class PygameSimulation(Simulation):
    """
    An interactive version of the Simulation class using Pygame.
    """
    # Colors
    COLOR_BG = (30, 30, 30)
    COLOR_OBSTACLE = (200, 50, 50)
    COLOR_ROBOT = (50, 200, 255)
    COLOR_PATH = (100, 255, 100)
    COLOR_TRAIL = (255, 255, 0)
    COLOR_GOAL = (50, 255, 50)
    """
    An interactive version of the Simulation class using Pygame.
    """
    # Colors
    COLOR_BG = (30, 30, 30)
    COLOR_OBSTACLE = (200, 50, 50)
    COLOR_ROBOT = (50, 200, 255)
    COLOR_PATH = (100, 255, 100)
    COLOR_TRAIL = (255, 255, 0)
    COLOR_GOAL = (50, 255, 50)
    
    def __init__(self, config_path="Scene/Configuration.json", params_path="Scene/Parameters.json"):
        super().__init__(config_path, params_path)
        
        # 1. Define Screen Size (But don't open it yet)
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1200, 800
        
        # 2. Calculate Scale Factors (Math only, no Pygame needed yet)
        map_w = self.world_bounds[0][1] - self.world_bounds[0][0]
        map_h = self.world_bounds[1][1] - self.world_bounds[1][0]
        
        self.scale_x = (self.SCREEN_WIDTH * 0.9) / map_w
        self.scale_y = (self.SCREEN_HEIGHT * 0.9) / map_h
        self.pixels_per_meter = min(self.scale_x, self.scale_y)
        
        self.offset_x = (self.SCREEN_WIDTH - (map_w * self.pixels_per_meter)) / 2
        self.offset_y = (self.SCREEN_HEIGHT - (map_h * self.pixels_per_meter)) / 2

        # Runtime State
        self.running = True
        self.paused = False
        self.sim_finished = False
        self.executed_states = []
        self.executed_controls = []
        
        # Placeholders for Pygame objects
        self.screen = None
        self.clock = None
        self.font = None
        
    def init_pygame_window(self):
        """Initializes the actual window and Pygame context."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Robot Control Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
    def world_to_screen(self, x: float, y: float):
        screen_x = int(self.offset_x + x * self.pixels_per_meter)
        screen_y = int(self.SCREEN_HEIGHT - (self.offset_y + y * self.pixels_per_meter))
        return screen_x, screen_y
    
    def draw_reference_point(self, ref_state):
        """Draws a crosshair at the target point the controller is chasing."""
        if ref_state is None: return
        
        # ref_state is [x_ref, y_ref, theta_ref, ...]
        # We only need x and y for the crosshair
        rx, ry = ref_state[0], ref_state[1]
        cx, cy = self.world_to_screen(rx, ry)
        
        # Draw a Magenta Crosshair
        size = 12
        color = (255, 0, 255) # Magenta
        pygame.draw.line(self.screen, color, (cx - size, cy), (cx + size, cy), 2)
        pygame.draw.line(self.screen, color, (cx, cy - size), (cx, cy + size), 2)
        # Optional: Draw a small direction line for theta_ref
        if len(ref_state) > 2:
            theta_ref = ref_state[2]
            end_x = rx + 0.5 * math.cos(theta_ref)
            end_y = ry + 0.5 * math.sin(theta_ref)
            ex, ey = self.world_to_screen(end_x, end_y)
            pygame.draw.line(self.screen, color, (cx, cy), (ex, ey), 1)
    
    def draw_robot(self, state):
        """Draws the robot body, heading arrow, and 4 steerable wheels with outlines."""
        x, y, theta = state[0], state[1], state[2]
        
        # --- 1. Robot Geometry ---
        L = self.robot_true.length
        W = self.robot_true.width
        wheel_radius = self.robot_true.wheel_radius
        
        # Wheel Dimensions
        wheel_len = 10 * wheel_radius 
        wheel_wid = wheel_len * 0.5 
        
        # Convert to pixels
        body_len_px = L * self.pixels_per_meter
        body_wid_px = W * self.pixels_per_meter
        w_len_px = wheel_len * self.pixels_per_meter
        w_wid_px = wheel_wid * self.pixels_per_meter
        
        # --- 2. Draw Body ---
        robot_surf = pygame.Surface((body_len_px, body_wid_px), pygame.SRCALPHA)
        pygame.draw.rect(robot_surf, self.COLOR_ROBOT, (0, 0, body_len_px, body_wid_px), border_radius=3)
        
        # --- Center & Heading Arrow ---
        ARROW_COLOR = (255, 80, 0)   # Bright Orange-Red
        cx, cy = body_len_px / 2, body_wid_px / 2
        
        # Center Dot
        pygame.draw.circle(robot_surf, ARROW_COLOR, (cx, cy), 6)
        
        # Heading Line (Arrow) - 75% length
        arrow_len_from_center = (body_len_px / 2) * 0.75
        end_x = cx + arrow_len_from_center
        pygame.draw.line(robot_surf, ARROW_COLOR, (cx, cy), (end_x, cy), 4)
        
        # Rotate and Blit Body
        rotated_body = pygame.transform.rotate(robot_surf, math.degrees(theta))
        body_rect = rotated_body.get_rect(center=self.world_to_screen(x, y))
        self.screen.blit(rotated_body, body_rect)

        # --- 3. Draw Wheels (High Contrast) ---
        # Surface for the wheel
        wheel_surf = pygame.Surface((w_len_px, w_wid_px), pygame.SRCALPHA)
        
        # Draw Black Wheel
        pygame.draw.rect(wheel_surf, (0, 0, 0), (0, 0, w_len_px, w_wid_px), border_radius=2)
        # Draw White Outline (2px thick) for visibility
        pygame.draw.rect(wheel_surf, (255, 255, 255), (0, 0, w_len_px, w_wid_px), width=2, border_radius=2)

        # Wheel offsets relative to robot center
        corners = [
            (L/2, W/2),   # Front Left
            (L/2, -W/2),  # Front Right
            (-L/2, W/2),  # Rear Left
            (-L/2, -W/2)  # Rear Right
        ]
        
        current_wheel_angles = state[6:10] if len(state) >= 10 else [0.0]*4

        for i, (dx_body, dy_body) in enumerate(corners):
            # Calculate Wheel Center in World Frame
            wx = x + dx_body * math.cos(theta) - dy_body * math.sin(theta)
            wy = y + dx_body * math.sin(theta) + dy_body * math.cos(theta)
            
            screen_pos = self.world_to_screen(wx, wy)

            # Calculate Total Wheel Angle
            steering_angle = current_wheel_angles[i]
            total_angle = theta + steering_angle
            
            # Rotate and Blit Wheel
            rotated_wheel = pygame.transform.rotate(wheel_surf, math.degrees(total_angle))
            wheel_rect = rotated_wheel.get_rect(center=screen_pos)
            self.screen.blit(rotated_wheel, wheel_rect)
    
    def draw_map(self):
        for obstacle in self.map_obj.obstacles:
            if obstacle.shape == ConvexShape.Circle:
                cx, cy = self.world_to_screen(obstacle.center[0], obstacle.center[1]) # type: ignore
                radius_px = int(obstacle.radius * self.pixels_per_meter) # type: ignore
                pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, (cx, cy), radius_px)
            elif obstacle.shape == ConvexShape.Polygon:
                points_px = [self.world_to_screen(p[0], p[1]) for p in obstacle.points]
                pygame.draw.polygon(self.screen, self.COLOR_OBSTACLE, points_px)

        gx, gy = self.world_to_screen(self.goal[0], self.goal[1])
        pygame.draw.circle(self.screen, self.COLOR_GOAL, (gx, gy), 10)

    def draw_path(self, path):
        if path is None or len(path) < 2: return
        screen_points = [self.world_to_screen(p[0], p[1]) for p in path]
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, screen_points, 2)

    def draw_trail(self):
        if len(self.executed_states) < 2: return
        # Optimization: step by 5 to save performance
        screen_points = [self.world_to_screen(s[0], s[1]) for s in self.executed_states[::5]]
        if len(screen_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRAIL, False, screen_points, 1)
    
    def draw_wind_field(self, disturbance_force):
        """
        Draws a grid of arrows representing the dynamic wind force.
        
        Args:
            disturbance_force: A tuple or array (fx, fy, torque)
        """
        fx, fy = disturbance_force[0], disturbance_force[1]
        magnitude = math.hypot(fx, fy)
        
        # 1. Threshold: Don't draw if wind is negligible
        if magnitude < 0.1:
            return

        # 2. Get Max Possible Force from params for scaling
        noise_params = self.params["Noise"]["Dynamic Disturbance"]
        max_x = noise_params["Max x Force"]
        max_y = noise_params["Max y Force"]
        # Approximate max magnitude (conservative estimate)
        max_mag = math.hypot(max_x, max_y)
        
        if max_mag == 0: return

        # 3. Calculate Arrow Length
        # Max length = 0.5 * Robot Length
        max_arrow_len_m = self.robot_true.length * 0.5
        # Current length scales with force intensity
        current_arrow_len_m = (magnitude / max_mag) * max_arrow_len_m
        
        # Convert to pixels
        arrow_len_px = current_arrow_len_m * self.pixels_per_meter
        
        # 4. Calculate Angle
        angle = math.atan2(fy, fx)
        
        # 5. Grid Settings
        # Draw an arrow every 10 meters to avoid clutter
        step_x_m = int(self.config["Map"]["Dimensions"]["Length"]/10)
        step_y_m = int(self.config["Map"]["Dimensions"]["Width"]/10)
        
        start_x, end_x = self.world_bounds[0]
        start_y, end_y = self.world_bounds[1]
        
        # Create a light semi-transparent color for wind (Cyan/White)
        wind_color = (200, 255, 255, 100) # RGBA
        
        # 6. Iterate and Draw
        current_x = start_x
        while current_x < end_x:
            current_y = start_y
            while current_y < end_y:
                cx, cy = self.world_to_screen(current_x, current_y)
                
                # Calculate end point of the arrow
                # Note: Screen Y is flipped, so we subtract dy
                ex = cx + arrow_len_px * math.cos(angle)
                ey = cy - arrow_len_px * math.sin(angle) 
                
                # Draw Line
                pygame.draw.line(self.screen, wind_color, (cx, cy), (ex, ey), 1)
                
                # Draw Arrowhead (Simple dot or small lines)
                pygame.draw.circle(self.screen, wind_color, (int(ex), int(ey)), 2)
                
                current_y += step_y_m
            current_x += step_x_m
            
    def run_interactive(self):        
        path = self.plan_path()
        if path is None: return

        self.init_pygame_window()
        print("Initializing Interactive Simulation...")        
        self.controller = self.get_controller(path)
        
        robot_sim = Robot_Sim(self.actuator_controller_true, self.robot_true, dt=self.dt)

        current_state = np.array([self.start[0], self.start[1], self.start[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot_sim.set_state(current_state)
        self.executed_states.append(current_state.copy())
        
        step = 0
        ref_state = None
        
        # Wind Timer
        noise_params = self.params["Noise"]
        wind_change_dt: float = noise_params["Dynamic Disturbance"]["Change dt"]
        steps_per_wind_change = int(wind_change_dt / self.dt)
        if steps_per_wind_change < 1: steps_per_wind_change = 1
        current_disturbance = np.zeros(3) 

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE: self.paused = not self.paused
                        elif event.key == pygame.K_ESCAPE: self.running = False

                if not self.paused and not self.sim_finished:
                    # Measurement
                    if self.state_uncertainty["Enable"]:
                        state_noise = add_state_estimation_uncertainty(
                            self.state_uncertainty["Position Noise StdDev"], 
                            self.state_uncertainty["Orientation Noise StdDev"], 
                            self.state_uncertainty["Linear Velocity Noise StdDev"], 
                            self.state_uncertainty["Angular Velocity Noise StdDev"])
                        measured_state = current_state + state_noise
                    else:
                        measured_state = current_state.copy()

                    ref_state = self.controller.get_reference_state(measured_state[:3])
                    control_input = self.controller.get_command(measured_state)
                    self.executed_controls.append(control_input.copy())
                    
                    # Wind Logic
                    if step % steps_per_wind_change == 0:
                        if noise_params["Dynamic Disturbance"]["Enable"]:
                            current_disturbance = add_force_uncertainty(
                                noise_params["Dynamic Disturbance"]["Max x Force"],
                                noise_params["Dynamic Disturbance"]["Max y Force"], 
                                noise_params["Dynamic Disturbance"]["Max Torque"])
                        else:
                            current_disturbance = np.zeros(3)
                    
                    # Propagate
                    new_state = robot_sim.propagate(current_state, control_input, noise=current_disturbance)
                    current_state = new_state
                    self.executed_states.append(current_state.copy())
                    step += 1

                    if self.controller.is_stabilized(current_state):
                        self.sim_finished = True

                # Rendering
                self.screen.fill(self.COLOR_BG)
                self.draw_map()
                self.draw_wind_field(current_disturbance)
                self.draw_path(path)
                self.draw_trail()
                if ref_state is not None:
                    self.draw_reference_point(ref_state)
                self.draw_robot(current_state)
                
                # HUD
                ui_text = [
                    f"Time: {step * self.dt:.2f}s",
                    f"Vel: {np.linalg.norm(current_state[3:5]):.2f} m/s",
                ]
                for i, line in enumerate(ui_text):
                    text_surf = self.font.render(line, True, (255, 255, 255))
                    self.screen.blit(text_surf, (15, 15 + i * 25))

                pygame.display.flip()
                self.clock.tick(60)

        finally:
            # --- FIXED: Save Screenshot BEFORE Quitting ---
            print("Saving Pygame screenshot...")
            pygame.image.save(self.screen, "pygame_final_state.png")
            print("Saved to 'pygame_final_state.png'")
            
            pygame.quit()
        
        # Plotting
        print("Generating error plots...")
        if len(self.executed_states) > 1:
            exec_states_np = np.array(self.executed_states)
            error_x, error_y, error_theta = self.calculate_path_errors(exec_states_np, path)
            cross_track_errors = self.calculate_cross_track_error(exec_states_np, path)
            self.plot_velocities_and_errors(exec_states_np, error_x, error_y, error_theta, cross_track_errors)

if __name__ == "__main__":
    sim = PygameSimulation()
    sim.run_interactive()