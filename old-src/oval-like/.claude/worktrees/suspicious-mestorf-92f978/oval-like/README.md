# Oval-Like Billiard Simulation

Simulates the dynamics of a billiard ball in an oval-like (non-circular) closed boundary. The boundary shape is defined in polar coordinates as:

$$r(\theta) = \frac{r_0}{1 + \varepsilon \cos(m\theta)}$$

where `r₀` is the base radius, `ε` (epsilon) is the eccentricity controlling boundary deformation, and `m` is the mode number controlling the number of lobes. The code computes trajectories, Poincaré sections (phase space), and Lyapunov exponents as a function of `ε`.

This is a Python port of a Fortran program originally written by João Pedro Cruz Ferreira.

---

## Setup

Requires Python ≥ 3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

To use the Jupyter notebooks, also install Jupyter:

```bash
uv pip install jupyter
```

---

## File Overview

### Core Library

#### [`billiard_engine.py`](billiard_engine.py)

The central simulation library. Contains:

- **`BilliardsNonTimeDependent`** — the main billiard class. Key methods:
  - `single_trajectory(theta_initial, alpha_initial, max_iterations)` — simulates one trajectory from a starting position `(θ, α)` on the boundary.
  - `simulate(...)` — samples a grid of initial conditions and returns a full Poincaré section as a DataFrame.
  - `get_convergence_data(theta0, alpha0, n_iters)` — computes the running Lyapunov exponent and its standard error using the analytical Jacobian and Welford's algorithm for numerical stability.
  - `jacobian(theta_n, theta_n1, alpha_n)` — analytical 2×2 Jacobian of the billiard map at a collision.
  - `plot_boundary`, `plot_phase_space`, `plot_trajectories` — matplotlib helpers.

- **`run_convergence_task(args)`** — a top-level picklable wrapper around `get_convergence_data`, suitable for use with `ProcessPoolExecutor`.

#### [`telemetry.py`](telemetry.py)

Progress-reporting utilities for long-running parallel simulations:

- **`Heartbeat`** — written to by each worker process; records its current tick to a file in `.progress/`.
- **`Monitor`** — read by the main process; aggregates all heartbeat files to report overall progress without inter-process communication overhead.

---

### Command-Line Scripts

#### [`run_lyapunov_sim.py`](run_lyapunov_sim.py)

Runs Lyapunov exponent convergence simulations for a fixed `(ε, m)` using several hardcoded initial conditions in parallel. Saves results to `billiard_results.npz` for later visualization.

```bash
uv run python run_lyapunov_sim.py
```

Configuration is edited directly in the script (parameters: `m=3`, `epsilon=0.1`, `n_iters=1000`). Output is an `.npz` file with convergence arrays for each seed.

---

#### [`run_epsilon_sweep.py`](run_epsilon_sweep.py)

Sweeps `ε` over a range of values, computes the converged Lyapunov exponent at each value, and saves the results to CSV. Uses `ProcessPoolExecutor` for parallelism on a single machine, with a live `tqdm` progress bar fed by the `telemetry` module.

```bash
uv run python run_epsilon_sweep.py \
    --min 0.001 \
    --max 0.5 \
    --steps 50 \
    --iters 100000 \
    --mode 3 \
    --output epsilon_sweep.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--min` | `0.001` | Minimum ε value |
| `--max` | `0.5` | Maximum ε value |
| `--steps` | `50` | Number of ε values to sweep |
| `--iters` | `1000` | Iterations per ε value |
| `--mode` | `3` | Mode number `m` (number of lobes) |
| `--output` | `epsilon_sweep.csv` | Output CSV filename |

Output CSV columns: `epsilon`, `lyapunov`, `error`, `m`, `iterations`.

---

#### [`run_epsilon_sweep_hpc.py`](run_epsilon_sweep_hpc.py)

The HPC-compatible version of the epsilon sweep. Instead of running all ε values in parallel locally, it processes a single ε value determined by the `SLURM_ARRAY_TASK_ID` environment variable. Designed to be called once per Slurm array task.

```bash
# Normally invoked by the Slurm scripts, but can also be run manually:
SLURM_ARRAY_TASK_ID=5 SLURM_ARRAY_TASK_COUNT=32 uv run python run_epsilon_sweep_hpc.py \
    --steps 32 \
    --iters 10000000 \
    --mode 3 \
    --min 0.001 \
    --max 0.05 \
    --outdir results/
```

Each task writes one output file: `results/epsilon_sweep_results_<task_id>.csv`.

---

#### [`merge_results.py`](merge_results.py)

Merges the per-task CSV files produced by `run_epsilon_sweep_hpc.py` into a single sorted CSV. Designed to run as a post-processing step after a Slurm array job completes.

```bash
uv run python merge_results.py --outdir runs/run_20260505_120000/data
```

Expects files matching `epsilon_sweep_results_*.csv` in `--outdir`. Writes a single merged file named `epsilon_sweep_results_<timestamp>.csv` in the same directory.

---

#### [`main.py`](main.py)

A minimal placeholder entry point. Currently just prints "Hello from python!".

---

### Slurm / HPC Scripts

These scripts automate running large-scale sweeps on Slurm clusters.

#### [`submit_slurm_script.sh`](submit_slurm_script.sh)

The top-level submission script. Run this from the cluster login node to launch a complete epsilon sweep and automatic result collation.

```bash
bash submit_slurm_script.sh
```

What it does:
1. Creates a timestamped run directory (`runs/run_YYYYMMDD_HHMMSS/data` and `logs/`).
2. Submits `slurm_epsilon_sweep.sh` as a Slurm array job and captures the job ID.
3. Submits `merge_results.py` as a dependent job (`afterok`), so it only runs if the sweep finishes successfully.

#### [`slurm_epsilon_sweep.sh`](slurm_epsilon_sweep.sh)

The Slurm batch script for the array job. Each task runs one instance of `run_epsilon_sweep_hpc.py` for a single ε value.

Default configuration (edit directly in the file):
- Array: tasks `0–31` (32 tasks)
- Iterations: `10,000,000` per ε value
- Mode: `m = 3`
- ε range: `[0.001, 0.05]`
- Memory: 4 GB per task
- Wall time: 24 hours per task

---

### Jupyter Notebooks

#### [`irregular_boundary_billiard.ipynb`](irregular_boundary_billiard.ipynb)

An introductory notebook that explains the physical system, derives the boundary equation, and demonstrates:
- Simulating billiard trajectories
- Plotting the billiard boundary and particle paths
- Generating and visualizing the Poincaré section (phase space)

Good starting point for understanding the simulation.

---

#### [`TrajectoryPhaseSpace.ipynb`](TrajectoryPhaseSpace.ipynb)

Generates trajectory plots and phase-space (Poincaré section) visualizations. Lets you interactively set `ε`, the mode `m`, and initial conditions `(θ₀, α₀)` to explore how the billiard dynamics change. Calls `BilliardsNonTimeDependent.simulate()` from the engine.

---

#### [`Lyapunov_Exponents.ipynb`](Lyapunov_Exponents.ipynb)

Loads the `.npz` output of `run_lyapunov_sim.py` and plots Lyapunov exponent convergence curves for each initial condition seed. Shows how `λ` stabilizes as the number of iterations grows, and overlays the mean across seeds.

---

#### [`EpsilonVLambda.ipynb`](EpsilonVLambda.ipynb)

Visualizes the relationship between eccentricity `ε` and the Lyapunov exponent `λ`. Loads a CSV produced by `run_epsilon_sweep.py` or `merge_results.py` and produces a publication-style plot showing how chaos (measured by `λ`) grows with `ε`.

Expects input at `data/epsilon_sweep_<timestamp>_<iters>iters.csv`.

---

### Tests

#### [`test_billiard_engine.py`](test_billiard_engine.py)

Pytest suite for the billiard engine. Contains:
- `test_jacobian_accuracy` — validates the analytical Jacobian against a finite-difference numerical approximation.
- `test_jacobian_determinant` — checks that the billiard map is area-preserving (Jacobian determinant = 1).

```bash
uv run pytest test_billiard_engine.py
```

---

## Typical Workflows

### Local exploration (single machine)

```bash
# 1. Run a quick Lyapunov convergence check for fixed (ε, m)
uv run python run_lyapunov_sim.py

# 2. Open the convergence plot notebook
jupyter notebook Lyapunov_Exponents.ipynb

# 3. Run an epsilon sweep
uv run python run_epsilon_sweep.py --steps 32 --iters 50000

# 4. Visualize ε vs λ
jupyter notebook EpsilonVLambda.ipynb
```

### High-resolution sweep on a Slurm cluster

```bash
# From the cluster login node:
bash submit_slurm_script.sh

# Monitor job status:
squeue -u $USER

# After completion, the merged CSV is in runs/run_<timestamp>/data/
# Copy it locally and open EpsilonVLambda.ipynb
```

---

## Parameters Reference

| Parameter | Symbol | Typical range | Effect |
|-----------|--------|---------------|--------|
| Base radius | `r₀` | `1.0` (fixed) | Overall scale |
| Eccentricity | `ε` | `0.001 – 0.5` | Boundary deformation; higher values increase chaos |
| Mode number | `m` | `2, 3, 4, ...` | Number of lobes; `m=3` gives a triangular-ish shape |
| Initial angle | `θ₀` | `[0, 2π)` | Starting position on boundary |
| Initial incidence | `α₀` | `(0, π)` | Angle of the initial trajectory relative to the tangent |
