SwerveVehicle
=============

Overview
--------

This repository contains a swerve-drive vehicle simulation, path planners, and controllers for research and educational use.

Implemented Controllers:
- **SMC** (Sliding Mode Control)
- **MRAC** (Model Reference Adaptive Control)
- **LQR** (Linear Quadratic Regulator)
- **ASMC** (Adaptive Sliding Mode Control)

Implemented Planners:
- **RRT\*** (Rapidly-exploring Random Tree Star)
- **A\*** (A Star)

---

Tuning Guide for Noisy Environments
-----------------------------------
When enabling noise or disturbances in `Parameters.json`, use these heuristics to maintain stability:

* **State Estimation Uncertainty:** If sensor noise is high, increase the `boundary_layer` (e.g., to 0.5+) and decrease the switching gains (`etas`) to prevent "chatter" and over-correction.
* **Dynamic Disturbances:** If external forces (e.g., wind or friction) are enabled, increase the `etas` gain. The robust term must be larger than the maximum expected disturbance to maintain path tracking.
* **Parameter Uncertainty:** When mass or inertia is unknown, the **ASMC** is the recommended controller. Set a small `gammas` (e.g., 0.01) to allow the system to learn the mass drift slowly without inducing instability.
* **Lookahead Smoothing:** For very noisy environments, increasing the `lookahead` distance in **PurePursuit** (e.g., to 3.0 or 4.0) acts as a geometric low-pass filter, resulting in smoother trajectories.


### Usage Recommendation
> **Note:** Use the **ASMC** controller if **both** `Parameter Uncertainty` and `Dynamic Disturbance` are set to `true` in `Parameters.json`. While standard SMC can handle disturbances, only ASMC can effectively compensate for a mismatched mass matrix, ensuring the robot doesn't undershoot or overshoot due to incorrect model assumptions.
> * If using no noise - use the LQR controller
> * If using only parametric uncertainties - use the MRAC controller
> * If using only Dynamic Disturbance: Use the SMC Controller

---

Controller Mathematical Analysis
---------------------------------------------

### 1. Sliding Mode Control (SMC)
The SMC is a robust tracking controller designed to drive the system to a predefined "sliding surface" where errors decay exponentially.

* **Sliding Surface ($s$):** Defined as $s = \dot{e} + \Lambda e$, where $e = x_{ref} - x$.
* **Equivalent Control ($u_{eq}$):** $u_{eq} = \dot{v}_{ref} + \Lambda \dot{e}$. For constant speed segments, $\dot{v}_{ref} \approx 0$.
* **Switching Control ($u_{sw}$):** $u_{sw} = K \cdot \tanh(s / \phi)$, where $\phi$ is the boundary layer to prevent chattering.
* **Lyapunov Stability:** Choosing $V = \frac{1}{2}s^T s$ ensures $\dot{V} = s^T [ -K \cdot \tanh(s/\phi) + \tau_d ]$. If $K > |\tau_d|_{max}$, then $\dot{V} < 0$, guaranteeing reaching the surface.

### 2. Linear Quadratic Regulator (LQR)
An optimal state-feedback controller based on a linearized double-integrator model.

* **State-Space Model:** $\dot{x} = Ax + Bu$, where $A$ and $B$ represent a decoupled double-integrator system for $x, y, \theta$.
* **Cost Function:** $J = \int_{0}^{\infty} (e^T Q e + u^T R u) dt$.
* **Control Law:** $u = -K e$, where $K = R^{-1} B^T P$ and $P$ is solved via the Algebraic Riccati Equation.
* **Stability:** Inherently stable for $Q \geq 0$ and $R > 0$, with a quadratic Lyapunov function $V = e^T P e$.

### 3. Model Reference Adaptive Control (MRAC)
This controller uses an internal "Reference Model" ($x_m$) and adapts a gain ($\hat{\alpha}$) to match it.

* **Reference Model:** $\ddot{x}_m = -k_p(x_m - x_{ref}) - k_v(\dot{x}_m - \dot{x}_{ref})$.
* **Update Law:** $\dot{\hat{\alpha}} = -\gamma \cdot (x - x_m)$. The gain $\hat{\alpha}$ adjusts to minimize the model tracking error.
* **Control Law:** $u = u_{nominal} \cdot \hat{\alpha}$, scaling the nominal gains to compensate for plant uncertainties.

---

### 4. Adaptive Sliding Mode Control (ASMC) Analysis
The ASMC is designed to handle systems where both the physical parameters (mass/inertia) are unknown and external disturbances are present.

#### Mathematical Foundation
The robot dynamics follow the model $M\ddot{x} + \tau_d = u$, where $M$ is the unknown mass matrix and $\tau_d$ represents external disturbances. The controller defines a **Sliding Surface ($s$)** to couple position and velocity errors:
$$s = e_v + \Lambda e_p$$
Where $e_p$ and $e_v$ are position and velocity tracking errors, and $\Lambda$ is the surface slope.

#### Control Law
The control command $u$ is calculated as:
$$u = \hat{M}(\Lambda e_v) + K_d s + \eta \tanh(s/\phi)$$
* **Equivalent Control ($\hat{M}\Lambda e_v$):** Uses the current mass estimate to maintain the sliding motion.
* **Robust Feedback ($K_d s + \eta \tanh$):** Drives the system to the surface and suppresses disturbances.

#### Parameter Adaptation Law
The mass matrix estimate $\hat{M}$ is updated in real-time according to the following update law:
$$\dot{\hat{M}} = \Gamma \cdot \text{diag}(s \cdot (\Lambda e_v))$$
This allows the controller to minimize the tracking error by adjusting its internal model of the robot's inertia.

#### Lyapunov Stability Analysis
To prove stability, we define the Lyapunov candidate function:
$$V = \frac{1}{2} s^T M s + \frac{1}{2} \tilde{M}^T \Gamma^{-1} \tilde{M}$$
Where $\tilde{M} = M - \hat{M}$ is the estimation error. Differentiating with respect to time:
$$\dot{V} = s^T M \dot{s} + \tilde{M}^T \Gamma^{-1} \dot{\tilde{M}}$$
Substituting $\dot{s} = (\ddot{x}_{ref} - \ddot{x}) + \Lambda e_v$ and the dynamics $M\ddot{x} = u - \tau_d$:
$$\dot{V} = s^T [M(\Lambda e_v) - (u - \tau_d)] - \tilde{M}^T \Gamma^{-1} \dot{\hat{M}}$$
Inserting the control law and update law, the parameter terms cancel out, leaving:
$$\dot{V} = -s^T K_d s - s^T \eta \tanh(s/\phi) + s^T \tau_d$$
For $\eta > |\tau_d|_{max}$, we ensure $\dot{V} \leq 0$, guaranteeing asymptotic stability and convergence to the path.

---



Key runnable files
------------------
- `SimulateController.py` — runs the full planner + controller test harness (planning, controller simulation, visualization).
- `ActuatorController/Simulation.py` — GUI / dynamic simulation for the low-level actuator model (Swerve simulator).
- `PathController/PointFollowing/SMCPoint_Controller.py` — standalone point-following SMC controller simulation and interactive visualizer.
- `Scene/SceneGenerator.py` — generate obstacle scenes/maps and save them to Scene/Configuration.json for planners to use.
- `Scene/VisualizeFreePoints.py` — find maximally-spaced free points (start/goal candidates) and visualize them on the generated map.

Recommended environment
-----------------------
- Python 3.10+ (virtualenv recommended)
- Required packages (common): `numpy`, `matplotlib`, `tkinter`.

Quick setup
-----------
From the project root (where this README and `Testing.py` live):

PowerShell / cmd (recommended):

```powershell
# create + activate virtualenv (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
# or .\.venv\Scripts\activate    # cmd

pip install -r requirements.txt   # if you maintain one; otherwise install numpy, matplotlib
```

Running the main components
---------------------------

Run from the project root so package imports resolve correctly.

- Run the full planner + controller test (headless + plots):

```powershell
python Testing.py
```


- Run the actuator GUI / low-level dynamic sim (Tkinter + Matplotlib GUI):

```powershell
python -m ActuatorController.Simulation
# or
python ActuatorController\Simulation.py
```

This launches the actuator-level simulation and GUI for the swerve drive. Use this to test low-level control and actuator dynamics.


- Run the point-following SMC demo (interactive simulation & error plots):

```powershell
python -m PathController.PointFollowing.SMCPoint_Controller
# or
python PathController\PointFollowing\SMCPoint_Controller.py
```

- Generate obstacle maps/scenes:

```powershell
python Scene/SceneGenerator.py
```

This will create or update the map and obstacle configuration in `Scene/Configuration.json`.

- Visualize and select maximally-spaced free points (for start/goal selection):

```powershell
python Scene/VisualizeFreePoints.py
```

This script visualizes the current map and obstacles, and finds a set of maximally-spaced, collision-free points. Use these as candidate start/goal locations for planning experiments.

Troubleshooting
---------------
- If you see ModuleNotFoundError for project modules (e.g. `ActuatorController`, `Scene`, `Uncertainties`), make sure you run the script from the repository root so Python can resolve top-level packages.
- Alternatively, set the `PYTHONPATH` to the project root before running, e.g. (PowerShell):

```powershell
$env:PYTHONPATH = "$(Resolve-Path .)" + ";" + $env:PYTHONPATH
python Testing.py
```

- For GUI windows (Tkinter + Matplotlib) ensure your environment supports GUI display (running remotely without an X server will not show windows).

Parameters & Config
-------------------
- `Scene/Configuration.json` contains environment and robot hardware parameters (map dimensions, obstacles, robot mass, inertia, wheel properties, limits). Edit this file to change the simulated world or robot physical parameters.
- `Scene/Parameters.json` contains algorithmic and noise parameters used by the test harness (controller gains, planner settings, and uncertainty/noise configuration). Use this file to tune controllers, enable/disable disturbances, and adjust planner settings.

### Uncertainty Configuration

You can control which uncertainties (e.g., actuator noise, sensor noise, model mismatch) are active in the simulation by editing the `Uncertainties` section in `Scene/Parameters.json`. Each uncertainty can be enabled/disabled and its parameters adjusted. To change which uncertainties are active:

1. Open `Scene/Parameters.json`.
2. Find the `Uncertainties` section.
3. Set the relevant flags (e.g., `"enable_actuator_noise": true`) and adjust parameters as needed.
4. Save the file and rerun your simulation.

This allows you to test controller and planner robustness under different disturbance scenarios.
- Both files are JSON. Make a backup before changing them. If you want to run with an alternate config, either modify these files or pass a custom path when invoking scripts that accept a `config_path` argument.

Notes
-----
- Some scripts perform their own `sys.path` tweaks to help with running from subfolders; the recommended approach is to run from the project root.