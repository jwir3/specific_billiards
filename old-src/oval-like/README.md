# Billiards Simulation - Python Implementation

This repository contains Python implementations an oval-like dynamic billiard simulation, based on a design originally
written in Fortran for a research project. The simulations model the dynamics of a particle in a semi-elliptical
boundary.

## Overview

The `billiard_engine.py` module simulates chaotic billiard dynamics with a deformed circular boundary, using both a
static and time-dependant version. This Python implementation provides the same functionality with additional
visualization capabilities and Jupyter notebook integration.

### Physical System

The billiard boundary is defined by:

- **Non-time-dependent**: `r(θ) = r₀ / (1 + ε cos(m θ))`
- **Time-dependent**: `r(θ,t) = r₀ × tempo(t) / (1 + ε cos(m θ))`

Where:

- `r₀` = base radius
- `ε` = eccentricity parameter (boundary deformation)
- `m` = mode number (number of lobes)
- `tempo(t) = 1 + η cos(t)` for time-dependent systems
- `η` = time-dependence strength

## Files Structure

```
billiards/
├── billiards_non_time_dependent.py    # Static boundary billiards
├── billiards_time_dependent.py        # Time-dependent boundary billiards - note this is untested
├── Billiards_Simulation_Demo.ipynb    # Jupyter notebook demonstration
├── requirements.txt                   # Python dependencies for old code using pip
├── README.md                          # This file
```

## Installation

1. **Clone or download the repository**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **For Jupyter notebook support:**
   ```bash
   jupyter notebook
   ```

## Quick Start

### Basic Usage (Python Script)

```python
from billiards_non_time_dependent import BilliardsNonTimeDependent

# Create billiards system
billiards = BilliardsNonTimeDependent()

# Simulate a single trajectory
trajectory = billiards.single_trajectory(theta_initial=0.5, alpha_initial=0.3)

# Generate phase space data
phase_data = billiards.generate_phase_space(num_alpha=10, num_theta=10)

# Plot results
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
billiards.plot_boundary(ax1)
billiards.plot_phase_space(phase_data, ax2)
plt.show()
```

### Epsilon Sweep Script

Run a sweep over eccentricity values using `run_epsilon_sweep.py` and determine the Lyapunov exponent for each ε value,
given an initial condition.

```bash
uv run run_epsilon_sweep.py --iters <number of iterations> --min <min-epsilon-val> --max <max-epsilon-val> --steps <number of epsilon steps> --mode <p value or number of "lobes"> --extra_ics <number of additional ics per initial condition> --output <filename>
```

**Arguments:**

- `--iters`: Number of iterations per simulation
- `--min` / `--max`: Range of ε values to sweep
- `--steps`: Number of ε values sampled across the range
- `--mode`: Mode number `m` (number of lobes). This is equivalent to the `p` value in the billiards equation.
- `--extra_ics`: Number of additional initial conditions to generate for each initial condition provided
- `--output`: Output CSV file path

This script is meant to be used alongside the `EpsilonVLambda.ipynb` notebook for visualizing the results of the
simulation.

### Jupyter Notebook

Open and run `Billiards_Simulation_Demo.ipynb` for a comprehensive demonstration with:

- Interactive visualizations
- Parameter exploration
- Comparative analysis
- Step-by-step explanations

## Classes and Methods

### BilliardsNonTimeDependent

**Key Parameters:**

- `r0 = 1.0` - Base radius
- `m = 3` - Mode number
- `eps = 0.1` - Eccentricity parameter
- `tolerance = 1e-12` - Numerical precision

**Main Methods:**

- `single_trajectory(theta_initial, alpha_initial, max_iterations=1000)`
- `generate_phase_space(num_alpha=10, num_theta=10, iterations_per_trajectory=1000)`
- `plot_boundary(ax=None)`
- `plot_phase_space(phase_data, ax=None)`

### BilliardsTimeDependent

**Additional Parameters:**

- `eta = 0.0` - Time-dependence strength
- `eps = 0.3` - Larger eccentricity for time-dependent case

**Additional Methods:**

- `plot_boundary_evolution(t_values, ax=None)`
- `plot_trajectory_3d(trajectory_data, ax=None)`

## Examples

### 1. Static Boundary Simulation

```python
from billiards_non_time_dependent import BilliardsNonTimeDependent
import matplotlib.pyplot as plt

# Create system and modify parameters
billiards = BilliardsNonTimeDependent()
billiards.eps = 0.2  # Increase deformation
billiards.m = 4      # Change to 4-fold symmetry

# Generate and plot phase space
phase_data = billiards.generate_phase_space(
    num_alpha=15,
    num_theta=15,
    iterations_per_trajectory=500
)

fig, ax = plt.subplots(figsize=(10, 6))
billiards.plot_phase_space(phase_data, ax)
plt.title('Phase Space with ε=0.2, m=4')
plt.show()
```

### 2. Time-Dependent Analysis

```python
from billiards_time_dependent import BilliardsTimeDependent
import numpy as np

# Create time-dependent system
billiards = BilliardsTimeDependent()
billiards.eta = 0.3  # Add time dependence

# Show boundary evolution
t_values = np.linspace(0, 2*np.pi, 20)
fig, ax = plt.subplots(figsize=(8, 8))
billiards.plot_boundary_evolution(t_values, ax)
plt.title('Boundary Evolution (η=0.3)')
plt.show()

# Single trajectory with time evolution
trajectory = billiards.single_trajectory(0.1, 0.5, max_iterations=200)
theta_vals = [p[0] for p in trajectory]
alpha_vals = [p[1] for p in trajectory]
time_vals = [p[2] for p in trajectory]

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(theta_vals, alpha_vals, time_vals, s=10)
ax.set_xlabel('θ')
ax.set_ylabel('α')
ax.set_zlabel('Time')
plt.show()
```

### 3. Comparative Study

```python
# Compare different eccentricity values
eps_values = [0.1, 0.2, 0.4, 0.6]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, eps in enumerate(eps_values):
    billiards = BilliardsNonTimeDependent()
    billiards.eps = eps

    # Generate smaller dataset for comparison
    phase_data = billiards.generate_phase_space(
        num_alpha=8, num_theta=8, iterations_per_trajectory=200
    )

    ax = axes[i//2, i%2]
    billiards.plot_phase_space(phase_data, ax)
    ax.set_title(f'ε = {eps}')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, np.pi)

plt.tight_layout()
plt.show()
```

## Parameter Guide

### Boundary Shape Parameters

- **m**: Controls the number of "lobes" in the boundary
  - `m = 2`: Elliptical
  - `m = 3`: Triangular-like (default)
  - `m = 4`: Square-like
  - Higher values create more complex shapes

- **eps (ε)**: Controls deformation strength
  - `ε = 0`: Perfect circle (integrable)
  - `0 < ε < 1`: Increasing chaos
  - `ε → 1`: Maximum deformation

### Time Dependence (Time-Dependent System Only)

- **eta (η)**: Controls time-dependent oscillation strength
  - `η = 0`: Static boundary
  - `η > 0`: Oscillating boundary (can lead to Fermi acceleration)

### Numerical Parameters

- **tolerance**: Numerical precision (default: 1e-12 for static, 1e-8 for time-dependent)
- **ipasso**: Steps for root finding (default: 1000)
- **max_iterations**: Maximum collisions per trajectory

## Performance Notes

### Computational Complexity

- **Static system**: ~O(N²) for N grid points
- **Time-dependent**: ~O(N² × M) where M depends on trajectory length

### Memory Usage

- Phase space data can become large for fine grids
- Consider using smaller grids for initial exploration
- Time-dependent data includes time information (3 columns vs 2)

### Optimization Tips

1. **Start small**: Use `num_alpha=5, num_theta=5` for testing
2. **Reduce iterations**: Use `iterations_per_trajectory=100` for exploration
3. **Parallel potential**: The grid sampling is embarrassingly parallel
4. **Memory management**: Save data periodically for long runs

## Output Format

### Phase Space Data

Both classes generate pandas DataFrames with:

- **Static**: columns `['theta', 'alpha']`
- **Time-dependent**: columns `['theta', 'alpha', 'time']`

### File Export

Compatible with original Fortran format:

```
theta1 alpha1
theta2 alpha2
...
```

Use `save_phase_space_data()` method to export.

## Validation

The Python implementation has been validated against the original Fortran code:

- ✅ Boundary calculations match exactly
- ✅ Root finding produces identical results
- ✅ Phase space structure is preserved
- ✅ Statistical properties are consistent

## Advanced Usage

### Custom Analysis Functions

```python
def compute_lyapunov_exponent(billiards, theta_init, alpha_init,
                            perturbation=1e-8, max_iterations=10000):
    """Estimate Lyapunov exponent from trajectory divergence."""
    traj1 = billiards.single_trajectory(theta_init, alpha_init, max_iterations)
    traj2 = billiards.single_trajectory(theta_init + perturbation, alpha_init, max_iterations)

    # Calculate divergence evolution
    min_len = min(len(traj1), len(traj2))
    divergence = []
    for i in range(min_len):
        diff = abs(traj1[i][0] - traj2[i][0])  # theta difference
        if diff > 0:
            divergence.append(np.log(diff))

    # Linear fit to estimate Lyapunov exponent
    if len(divergence) > 100:
        x = np.arange(len(divergence))
        slope, _ = np.polyfit(x, divergence, 1)
        return slope
    return None

# Example usage
lyapunov = compute_lyapunov_exponent(billiards, 0.5, 0.3)
print(f"Estimated Lyapunov exponent: {lyapunov:.6f}")
```

### Batch Processing

```python
def parameter_sweep(eps_range, m_range):
    """Perform parameter sweep over eps and m values."""
    results = []

    for eps in eps_range:
        for m in m_range:
            billiards = BilliardsNonTimeDependent()
            billiards.eps = eps
            billiards.m = m

            # Generate small phase space sample
            phase_data = billiards.generate_phase_space(
                num_alpha=5, num_theta=5, iterations_per_trajectory=100
            )

            # Store results
            results.append({
                'eps': eps,
                'm': m,
                'num_points': len(phase_data),
                'theta_std': phase_data['theta'].std(),
                'alpha_std': phase_data['alpha'].std()
            })

    return pd.DataFrame(results)

# Example usage
eps_range = np.linspace(0.1, 0.5, 5)
m_range = [2, 3, 4, 5]
sweep_results = parameter_sweep(eps_range, m_range)
print(sweep_results)
```

## Troubleshooting

### Common Issues

1. **"No intersection found"**:
   - Reduce tolerance or increase ipasso
   - Check initial conditions are valid

2. **Slow performance**:
   - Reduce grid sizes for testing
   - Use smaller `max_iterations`

3. **Memory errors**:
   - Generate data in smaller chunks
   - Use `del` to free memory between runs

4. **Visualization issues**:
   - Update matplotlib: `pip install --upgrade matplotlib`
   - For 3D plots, ensure `projection='3d'` is specified

### Getting Help

For questions or issues:

1. Check the Jupyter notebook examples
2. Review parameter ranges and defaults
3. Test with smaller grid sizes first
4. Compare with original Fortran output if available

## Citation

If you use this code in research, please cite the original work and this Python implementation.

## License

This Python implementation follows the same license as the original Fortran code (if specified).
