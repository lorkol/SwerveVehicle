
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Arrow, Polygon
import time

# Add parent directory to sys.path for module imports
import sys
import os
from pathlib import Path

# Ensure project root (SwerveVehicle) is on sys.path.
# For this file (ActuatorController/Simulation.py) the project root is two levels up.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ActuatorController.ActuatorController import ActuatorController
from Scene.Robot import Robot

class SwerveSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Swerve Drive Physics Simulator")
        self.root.geometry("1400x900")

        # --- Simulation State ---
        self.running = False
        self.start_time = 0
        self.dt = 0.05  # Time step (seconds)
        
        # State Vector: [x, y, theta, vx_R, vy_R, v_theta]
        # Start robot in the middle of the map
        
        
        # History for plotting trails
        self.path_x = []
        self.path_y = []


        # --- Map Setup ---
        # Load map config from JSON
        import json
        map_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scene', 'Configuration.json'))
        with open(map_config_path, 'r') as f:
            config_json = json.load(f)
        self.map_config = {
            "Dimensions": {
                "Length": config_json["Map"]["Dimensions"]["Length"],
                "Width": config_json["Map"]["Dimensions"]["Width"]
            }
        }
        map_L = self.map_config["Dimensions"]["Length"]
        map_W = self.map_config["Dimensions"]["Width"]
        
        self.state = np.zeros(6)
        self.state[0] = map_L / 2  # x
        self.state[1] = map_W / 2  # y

        # --- Robot Setup ---
        # Default Robot Params
        self.robot_params = {
            "Dimensions": {"Length": 0.6, "Width": 0.6},
            "Mass": 30.0,
            "Wheel Radius": 0.05,
            "Inertia": 5.0,
            "Max Wheel Rotation Speed": 20.0,
            "Max Wheel Torque": 2.0,
            "Max Steering Speed": 10.0,
            "Max Steering Acceleration": 10.0,
            "Max Steering Torque": 10.0
        }
        self.robot = Robot(self.robot_params)
        self.controller = ActuatorController(self.robot)

        # --- GUI Layout ---
        # Main Container
        self.main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left: Controls
        self.control_frame = ttk.Frame(self.main_pane, width=400, padding="10")
        self.main_pane.add(self.control_frame)

        # Right: Visualization
        self.viz_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.viz_frame)

        # --- Visualization Setup ---
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        # Set map size from config
        map_L = self.map_config["Dimensions"]["Length"]
        map_W = self.map_config["Dimensions"]["Width"]
        self.ax.set_xlim(0, map_L)
        self.ax.set_ylim(0, map_W)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Robot Artists
        self.robot_rect = Rectangle((0,0), 0, 0, fill=False, edgecolor='blue', lw=2)
        self.ax.add_patch(self.robot_rect)
        self.heading_arrow = None # Will be updated
        self.trail_line, = self.ax.plot([], [], 'g--', alpha=0.5)

        # --- Controls Implementation ---
        self._build_controls()
        
        # Start Loop
        self.update_loop()

    def _build_controls(self):
        # 1. Simulation Control
        sim_box = ttk.LabelFrame(self.control_frame, text="Simulation Control", padding="5")
        sim_box.pack(fill=tk.X, pady=5)
        
        self.btn_play = ttk.Button(sim_box, text="Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = ttk.Button(sim_box, text="Reset", command=self.reset_sim)
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        self.lbl_status = ttk.Label(sim_box, text="Status: Paused", foreground="red")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # 2. Robot Parameters (Live Update)
        param_box = ttk.LabelFrame(self.control_frame, text="Robot Parameters", padding="5")
        param_box.pack(fill=tk.X, pady=5)
        
        self.sliders_params = {}
        self._add_slider(param_box, "Mass (kg)", 1, 100, self.robot_params["Mass"], "Mass")
        self._add_slider(param_box, "Inertia (kg*m^2)", 0.1, 20, self.robot_params["Inertia"], "Inertia")
        self._add_slider(param_box, "Wheel Radius (m)", 0.01, 0.2, self.robot_params["Wheel Radius"], "Wheel Radius")

        # 3. Inputs (Wheel Controls)
        input_box = ttk.LabelFrame(self.control_frame, text="Wheel Inputs", padding="5")
        input_box.pack(fill=tk.BOTH, expand=True, pady=5)

        # Notebook for Wheels
        self.notebook = ttk.Notebook(input_box)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.wheel_controls = []
        wheel_names = ["FR", "FL", "RL", "RR"]
        
        for i, name in enumerate(wheel_names):
            page = ttk.Frame(self.notebook)
            self.notebook.add(page, text=name)
            controls = {}
            # Steering Delta (degrees for UI, convert to radians internally)
            controls['delta'] = self._add_slider(page, f"Steering {name} (deg)", -720, 720, 0.0, None)
            # Torque (allow small torques)
            controls['torque'] = self._add_slider(page, f"Torque {name} (Nm)", -2.0, 2.0, 0.0, None)
            self.wheel_controls.append(controls)
        
        # Global Torque Scaler (allow small torques)
        self.global_torque_slider = self._add_slider(input_box, "All Wheels Torque", -2.0, 2.0, 0.0, None, command=self.update_all_torques)

        # Global Steering Angle Scaler (degrees, wide range)
        self.global_steering_slider = self._add_slider(input_box, "All Wheels Steering (deg)", -720, 720, 0.0, None, command=self.update_all_steering)

    def _add_slider(self, parent, label, min_val, max_val, init_val, param_key, command=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(frame, text=label, width=20)
        lbl.pack(side=tk.LEFT)

        var = tk.DoubleVar(value=init_val)

        # Numeric Entry
        entry = ttk.Entry(frame, textvariable=var, width=6)
        entry.pack(side=tk.RIGHT)

        # Scale
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # If this is a robot parameter, bind update
        if param_key:
            scale.configure(command=lambda v, k=param_key: self.update_robot_param(k, float(v)))
            entry.bind('<Return>', lambda e, k=param_key, v=var: self.update_robot_param(k, v.get()))
        elif command:
            scale.configure(command=command)
            # For wheel controls, update slider when entry is changed
            entry.bind('<Return>', lambda e, v=var: scale.set(v.get()))

        return var


    def update_all_torques(self, val):
        """Helper to set all wheel torques at once for easier testing"""
        val = float(val)
        for ctrl in self.wheel_controls:
            ctrl['torque'].set(val)

    def update_all_steering(self, val):
        """Helper to set all wheel steering angles at once for easier testing"""
        val = float(val)
        for ctrl in self.wheel_controls:
            ctrl['delta'].set(val)

    def update_robot_param(self, key, value):
        # Update internal dict
        if key in self.robot_params:
            self.robot_params[key] = value
        elif key in ["Length", "Width"]:
            self.robot_params["Dimensions"][key] = value
            
        # Re-init robot and controller to apply mass/inertia changes
        # Note: In a real efficient sim we'd just set properties, but this ensures 
        # the ActuatorController recalculates matrices like self._M_MATRIX
        self.robot = Robot(self.robot_params)
        self.controller = ActuatorController(self.robot)

    def toggle_play(self):
        self.running = not self.running
        if self.running:
            self.btn_play.configure(text="Pause")
            self.lbl_status.configure(text="Status: Running", foreground="green")
        else:
            self.btn_play.configure(text="Play")
            self.lbl_status.configure(text="Status: Paused", foreground="red")

    def reset_sim(self):
        self.running = False
        self.btn_play.configure(text="Play")
        self.lbl_status.configure(text="Status: Paused", foreground="red")
        # Reset state to center of map
        map_L = self.map_config["Dimensions"]["Length"]
        map_W = self.map_config["Dimensions"]["Width"]
        self.state = np.zeros(6)
        self.state[0] = map_L / 2  # x
        self.state[1] = map_W / 2  # y
        self.path_x = []
        self.path_y = []
        self.draw_robot()

    def get_inputs(self):
        # Extract angles (convert from degrees to radians) and torques from sliders
        angles = []
        torques = []
        for ctrl in self.wheel_controls:
            angles.append(np.deg2rad(ctrl['delta'].get()))
            torques.append(ctrl['torque'].get())
        return np.array(angles), np.array(torques)

    def update_physics(self):
        if not self.running:
            return

        wheel_angles, wheel_torques = self.get_inputs()
        
        # Get Derivatives using the provided Controller logic
        # dXdt = [vx_G, vy_G, v_theta, ax_R, ay_R, a_theta]
        dXdt = self.controller.get_state_derivatives(self.state, wheel_angles, wheel_torques)
        
        # Euler Integration
        # Position += Velocity * dt
        self.state[0] += dXdt[0] * self.dt # x
        self.state[1] += dXdt[1] * self.dt # y
        self.state[2] += dXdt[2] * self.dt # theta
        
        # Velocity (Robot Frame) += Acceleration * dt
        self.state[3] += dXdt[3] * self.dt # vx_R
        self.state[4] += dXdt[4] * self.dt # vy_R
        self.state[5] += dXdt[5] * self.dt # v_theta

        # Store path
        self.path_x.append(self.state[0])
        self.path_y.append(self.state[1])
        
        # Trim path to keep memory low
        if len(self.path_x) > 1000:
            self.path_x.pop(0)
            self.path_y.pop(0)

    def draw_robot(self):
        x, y, theta = self.state[0], self.state[1], self.state[2]
        L = self.robot_params["Dimensions"]["Length"]
        W = self.robot_params["Dimensions"]["Width"]

        # Calculate corners for rotation
        # Center is x,y. 
        # Local corners: (+L/2, +W/2), etc.
        # NOTE: Your ActuatorController uses L and W as half-lengths relative to center?
        # Checking ActuatorController: wheel positions are [self._l, -self._w].
        # Usually standard is L=Length, l=L/2. 
        # Let's assume the parameters passed to Robot are FULL dimensions, 
        # but the ActuatorController might store them differently. 
        # Robot.py: self.length = json["Length"].
        # ActuatorController.py: self._l = robot.length.
        # IF ActuatorController treats robot.length as the distance from center (half-length), 
        # then the drawing size should be 2*L. 
        # IF ActuatorController treats it as full length, it calculates torque with full length as arm.
        # Based on: `arm_length = math.sqrt(self._l**2 + self._w**2)` in init, 
        # it seems ActuatorController treats .length as the distance from center (radius-like).
        
        # Drawing assuming .length is distance from center (Half-Length)
        # because ActuatorController uses it directly for torque arms.
        
        l_draw = self.robot.length 
        w_draw = self.robot.width
        
        # Rotation Matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Corners relative to center
        # FR, FL, RL, RR
        corners = np.array([
            [l_draw, -w_draw],
            [l_draw, w_draw],
            [-l_draw, w_draw],
            [-l_draw, -w_draw]
        ])
        
        # Rotate and translate
        rot_corners = corners @ R.T + np.array([x, y])
        
        # Update Rectangle (using a polygon would be easier, but let's just set xy)
        # Matplotlib Rectangle takes bottom-left. We need to calculate it.
        # Actually, since it rotates, we can't use standard Rectangle easily without transforms.
        # Let's simple remove and redraw a polygon, or set transform.
        # Easiest for real-time: Clear and plot lines.
        
        self.robot_rect.remove()
        # Create a polygon for the chassis
        self.robot_rect = Polygon(rot_corners, closed=True, fill=False, edgecolor='blue', lw=2)
        self.ax.add_patch(self.robot_rect)
        
        # Draw Heading Arrow
        if self.heading_arrow:
            self.heading_arrow.remove()
        
        arrow_len = 0.5
        self.heading_arrow = Arrow(x, y, arrow_len*c, arrow_len*s, width=0.2, color='red')
        self.ax.add_patch(self.heading_arrow)
        
        # Update Trail
        self.trail_line.set_data(self.path_x, self.path_y)
        
        # Center view on robot if it goes off screen?
        # Or just keep fixed. Let's keep fixed but adjustable via toolbar (default matplotlib toolbar)
        
        self.canvas.draw()

    def update_loop(self):
        start = time.time()
        
        self.update_physics()
        self.draw_robot()
        
        # Calculate delay to maintain frame rate
        elapsed = time.time() - start
        delay = max(1, int((self.dt - elapsed) * 1000))
        
        self.root.after(delay, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SwerveSimulatorGUI(root)
    root.mainloop()