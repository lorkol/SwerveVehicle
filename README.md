SwerveVehicle
=============

Overview
--------
This repository contains a swerve-drive vehicle simulation, path planners, and controllers.

Key runnable files
------------------
- `Testing.py` — runs the full planner + controller test harness (planning, controller simulation, visualization).
- `ActuatorController/Simulation.py` — GUI / dynamic simulation for the low-level actuator model (Swerve simulator).
- `PathController/PointFollowing/SMCPoint_Controller.py` — standalone point-following SMC controller simulation and interactive visualizer.
- `Scene/SceneGenerator.py` — generate obstacle scenes/maps and save them to Scene/Configuration.json for planners to use.
- `Scene/VisualizeFreePoints.py` — find maximally-spaced free points (start/goal candidates) and visualize them on the generated map.

Recommended environment
-----------------------
- Python 3.10+ (virtualenv recommended)
- Required packages (common): `numpy`, `matplotlib`, `tkinter` (usually included with Python on Windows). If you use a virtualenv, install packages with pip.

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

- Run the point-following SMC demo (interactive simulation & error plots):

```powershell
python -m PathController.PointFollowing.SMCPoint_Controller
# or
python PathController\PointFollowing\SMCPoint_Controller.py
```

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
- Both files are JSON. Make a backup before changing them. If you want to run with an alternate config, either modify these files or pass a custom path when invoking scripts that accept a `config_path` argument.

Notes
-----
- Some scripts perform their own `sys.path` tweaks to help with running from subfolders; the recommended approach is to run from the project root.
- If you want a cleaner development setup, consider turning the repository into an installable package (`pip install -e .`) or adding a short launcher script that sets up `PYTHONPATH` and runs the selected module.

If you want, I can also:
- Add a `requirements.txt` with the detected dependencies.
- Add a tiny `run.sh` / `run.ps1` launcher that runs the common commands above.
- Convert relative imports and provide an editable pip install layout.
