# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "matplotlib",
# ]
# ///
#
# Above is the uv script metadata block so you can run this with:
# uv run billiard_engine.py
#
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from telemetry import Heartbeat


def angle_diff(a, b, period):
    """Calculates the shortest difference between two angles."""
    diff = a - b
    return (diff + period / 2) % period - period / 2


def run_patch_task(ecc, iters, extra_ics, progress_queue):
    """
    HPC/Multi-core task wrapper that executes a targeted patch run for a
    specific epsilon value. It pipes progress updates back through a shared
    multiprocessing queue and returns fully converged statistics.
    """
    # 1. Establish the standard experiment variables matching your baseline configuration
    scale = 1.0
    m = 3
    t0, a0 = 1.0, 0.5
    verbose = True  # Ensures status feedback maps cleanly to logs

    # 2. Reconstruct the exact 9-parameter tuple your engine's internal architecture handles:
    # (scale, eccentricity, m, iters, t0, a0, extra_ics, queue, verbose)
    task_tuple = (scale, ecc, m, iters, t0, a0, extra_ics, progress_queue, verbose)

    # 3. Delegate to your existing task runner to safely unpack and process the simulation
    _, avg_lambda_arr, avg_error_arr = run_convergence_task(task_tuple)

    # 4. Extract the final index values at the end of the full collision chain
    return ecc, avg_lambda_arr[-1], avg_error_arr[-1]


def run_convergence_task(args):
    r0, ecc, m, n_iters, t0, a0, num_extra_ics, progress_queue, should_output_logs = (
        args
    )

    hb = Heartbeat(progress_queue)

    sim = BilliardsNonTimeDependent(
        r0, ecc, m, num_extra_ics=num_extra_ics, should_output_logs=should_output_logs
    )
    return sim.get_convergence_data(t0, a0, n_iters, heartbeat=hb)


def run_history_task(args):
    """New task wrapper for parallel Lyapunov exponent history generation."""
    r0, ecc, m, n_iters, t0, a0, num_extra_ics, progress_queue, should_output_logs = (
        args
    )
    hb = Heartbeat(progress_queue)
    sim = BilliardsNonTimeDependent(r0, ecc, m, should_output_logs=should_output_logs)
    # Call the new history-focused method
    return sim.get_convergence_history(t0, a0, n_iters, heartbeat=hb)


class BilliardsNonTimeDependent:
    def __init__(
        self,
        r0=1.0,
        eccentricity=0.1,
        mode=3,
        num_extra_ics=0,
        should_output_logs=False,
    ):
        self.r0 = r0
        self.m = mode
        self.eccentricity = eccentricity
        self.num_extra_ics = num_extra_ics
        self.tolerance = 1e-12
        self.root_passes = 100
        self.trajectory_steps = 200
        self.bisection_iterations = 1000000
        self.grid_density = 10
        self.save_density = 2000
        self.should_output_logs = should_output_logs

        if self.should_output_logs and not os.path.exists("logs"):
            os.makedirs("logs")

    def _truncate_ic(self, theta, alpha, sig_figs=4):
        """
        Truncates values to a set number of significant figures to find a new IC in the same sea.

        Parameters
        ----------
        theta : float
            The angle of the initial condition to truncate.
        alpha : float
            The angle of the initial condition to truncate.
        sig_figs : int, optional
            The number of significant figures to truncate to (default is 4).

        Returns
        -------
        theta_trunc : float
            The truncated angle of the initial condition.
        alpha_trunc : float
            The truncated angle of the initial condition.
        """

        def trunc(x):
            if x == 0:
                return 0
            return np.round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

        return trunc(theta), trunc(alpha)

    def _generate_additional_initial_conditions(self, theta0, alpha0):
        """
        Generates a list of initial conditions based on the provided theta0 and alpha0, including any extra initial
        conditions if specified.
        """
        all_ics = [(theta0, alpha0)]
        if self.num_extra_ics > 0:
            # This is an amount that is guaranteed to be within the chaotic sea, if the original IC is already in
            # the chaotic sea.
            adjustment_factor = self.eccentricity / 100.0
            for i in range(1, self.num_extra_ics + 1):
                t_rand = theta0 + np.random.uniform(
                    -adjustment_factor, adjustment_factor
                )
                a_rand = alpha0 + np.random.uniform(
                    -adjustment_factor, adjustment_factor
                )
                all_ics.append((t_rand, a_rand))
        return all_ics

    def _run_convergence_for_initial_condition(
        self,
        curr_theta,
        curr_alpha,
        n_iters,
        log_interval,
        heartbeat,
        log_file,
        log_targets,
    ):
        history_n = []
        history_lam = []
        v = np.array([1.0, 0.0])
        running_mu = 0.0
        running_m2 = 0.0

        for i in range(1, n_iters + 1):
            base = self.step_map(curr_theta, curr_alpha)
            if not base:
                break

            t_next, _ = base
            J = self.jacobian(curr_theta, t_next, curr_alpha)

            v_next = J @ v
            norm = np.linalg.norm(v_next)

            # NOTE_SJJ: If the particle exhibits a catastrophic grazing collision, the norm will be basically
            # garbage (either nan or inf). In this case, we realign the vector to the physical boundary spatial
            # tangent (d_phi = 1.0) and add a tiny random perturbation to keep the subsequent float64 math stable.
            # If we don't do this, the subsequent calculations will overflow or underflow, leading to incorrect
            # results, especially at epsilon = 0.5 or other very extreme cases.
            # Originally, I had a `continue` here, but this caused even more instability. Instead of loop-breaking,
            # safely patch the norm and let Welford's algorithm handle a neutral (0) expansion step.
            if norm < 1e-14 or not np.isfinite(norm):
                v = np.array([1.0, np.random.uniform(-1e-4, 1e-4)])
                v /= np.linalg.norm(v)
                norm = 1.0  # ln(1.0) = 0.0 contribution
            else:
                v = v_next

            x = np.log(norm)

            # Welford's Algorithm
            delta_mean = x - running_mu
            running_mu += delta_mean / i
            running_m2 += delta_mean * (x - running_mu)

            v /= norm
            curr_theta, curr_alpha = base

            if heartbeat and i % 500 == 0:
                heartbeat.report(1)

            # Linear CSV Logging (Used by data sweep tracking)
            if log_file and i % log_interval == 0:
                current_var = running_m2 / (i - 1) if i > 1 else 0
                current_sem = np.sqrt(current_var / i)
                log_file.write(f"{i},{running_mu:.10e},{current_sem:.10e}\n")

            # Logarithmic History Sampling (Used for plotting convergence curves)
            # Only runs if we explicitly requested history arrays and pre-generated log_targets
            if log_targets and (i in log_targets):
                history_n.append(i)
                history_lam.append(running_mu)

        return (running_mu, running_m2, history_n, history_lam)

    ## TODO: Add note about this function and why it's here
    def get_convergence_history(
        self, theta0, alpha0, n_iters, num_points_desired=500, heartbeat=None
    ):
        curr_theta, curr_alpha = theta0, alpha0
        v = np.array([1.0, 0.0])
        running_mu = 0.0
        running_m2 = 0.0

        history_n = []
        history_lam = []

        log_targets = set(
            np.unique(
                np.logspace(0, np.log10(n_iters), num=num_points_desired, dtype=int)
            )
        )

        (running_mu, running_m2, h_n, h_lam) = (
            self._run_convergence_for_initial_condition(
                curr_theta, curr_alpha, n_iters, None, heartbeat, None, log_targets
            )
        )

        history_n.extend(h_n)
        history_lam.extend(h_lam)

        variance = running_m2 / (n_iters - 1) if n_iters > 1 else 0.0
        sem = np.sqrt(variance / n_iters)

        return np.array(history_n), np.array(history_lam), running_mu, sem

    def get_convergence_data(
        self, theta0, alpha0, n_iters=2, log_targets=None, heartbeat=None
    ):
        """
        Calculates the Lyapunov exponent with optional convergence logging.
        """
        all_ics = self._generate_additional_initial_conditions(theta0, alpha0)
        final_lambdas = []
        final_errors = []
        history_n = []
        history_lam = []

        # 2. Run convergence for every IC in the ensemble
        for idx, (ic_t, ic_a) in enumerate(all_ics):
            curr_theta, curr_alpha = ic_t, ic_a

            # Optional: Initialize unique log file for this specific IC and thread
            log_file = None
            num_points_desired = 200
            log_interval = max(1, n_iters // num_points_desired)
            if getattr(self, "should_output_logs", False):
                pid = os.getpid()
                log_filename = (
                    f"logs/conv_log_eps{self.eccentricity:.3f}_pid{pid}_ic{idx}.csv"
                )
                log_file = open(log_filename, "w")
                log_file.write("iteration,lambda,error\n")

            (running_mu, running_m2, h_n, h_lam) = (
                self._run_convergence_for_initial_condition(
                    curr_theta,
                    curr_alpha,
                    n_iters,
                    log_interval,
                    heartbeat,
                    log_file,
                    log_targets,
                )
            )

            # Only track history arrays for the core baseline initial condition (idx 0)
            if idx == 0:
                history_n.extend(h_n)
                history_lam.extend(h_lam)

            if log_file:
                log_file.close()

            final_lambdas.append(running_mu)
            if n_iters > 1:
                variance = running_m2 / (n_iters - 1)
                final_errors.append(np.sqrt(variance / n_iters))
            else:
                final_errors.append(0.0)

        if heartbeat:
            heartbeat.cleanup()

        # 3. Robust Averaging using Median
        avg_lambda = np.median(final_lambdas)
        avg_error = np.sqrt(np.sum(np.square(final_errors))) / len(final_errors)

        return np.array([n_iters]), np.array([avg_lambda]), np.array([avg_error])

    # These are from Diego's thesis.
    # def boundary_radius(self, theta):
    #     return self.r0 * (1.0 + self.eccentricity * np.cos(self.m * theta))

    # def boundary_derivative(self, theta):
    #     return -self.r0 * self.eccentricity * self.m * np.sin(self.m * theta)

    def boundary_radius(self, theta):
        """Calculate the boundary radius at angle theta."""
        return self.r0 / (1.0 + self.eccentricity * np.cos(self.m * theta))

    def boundary_derivative(self, theta):
        """Calculate the derivative of boundary radius with respect to theta."""
        return (self.r0 * self.eccentricity * self.m * np.sin(self.m * theta)) / (
            (1.0 + self.eccentricity * np.cos(self.m * theta)) ** 2
        )

    def cartesian_coords(self, theta):
        """
        Convert polar coordinates on the boundary to Cartesian.

        Because our boundary is defined between 0 and 2π in polar coordinates, we can find a point on the boundary
        in Cartesian coordinates by first finding the radius at the given value of theta and then using this to
        find the x and y coordinates.

        Parameters:
          - theta: The angle, in polar coordinates, of the ray where the boundary point should be found.

        Returns:
            A tuple containing the following values:
                - x: The x coordinate of the point.
                - y: The y coordinate of the point.
                - r: The radius of the billiard at the given theta value.
        """
        r = self.boundary_radius(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, r

    def boundary_tangent_angle(self, theta):
        """Calculate the angle of the tangent to the boundary at theta."""
        r = self.boundary_radius(theta)
        dr = self.boundary_derivative(theta)
        x, y, _ = self.cartesian_coords(theta)

        dx = dr * np.cos(theta) - y
        dy = dr * np.sin(theta) + x

        phi = np.arctan2(dy, dx)
        if phi < 0:
            phi += 2 * np.pi
        return phi

    def intersection_function(self, theta_test, x, y, alpha, phi):
        """
        Function to find intersection points with the boundary.
        Returns 0 when the ray from (x,y) at angle (alpha+phi) intersects the boundary.
        """
        r_test = self.boundary_radius(theta_test)
        return (r_test * np.sin(theta_test) - y) * np.cos(alpha + phi) - (
            r_test * np.cos(theta_test) - x
        ) * np.sin(alpha + phi)

    def find_intersections(self, x, y, alpha, phi):
        """Find all intersection points of a ray with the boundary using the bisection method."""
        solutions = []

        # Discretize the angle space to find sign changes
        theta_step = 2 * np.pi / self.root_passes

        for i in range(self.root_passes):
            a = i * theta_step
            b = (i + 1) * theta_step

            fa = self.intersection_function(a, x, y, alpha, phi)
            fb = self.intersection_function(b, x, y, alpha, phi)

            # Check for sign change (root exists in interval)
            if fa * fb < 0:
                try:
                    # Use bisection method to find root
                    root = brentq(
                        lambda theta: self.intersection_function(
                            theta, x, y, alpha, phi
                        ),
                        a,
                        b,
                        xtol=self.tolerance,
                        maxiter=self.bisection_iterations,
                    )
                    solutions.append(root)

                    if len(solutions) >= 4:  # Fortran code limits to 4 solutions
                        break
                except ValueError:
                    continue

        # Pad with zeros if needed (to match Fortran behavior)
        while len(solutions) < 4:
            solutions.append(0.0)

        return solutions[:4]

    def select_next_collision_simple(self, thetai, x, y, solutions):
        """Select next collision point when there are 2 or fewer valid solutions."""
        for i in range(2):
            if solutions[i] != 0:
                theta = solutions[i]
                xn, yn, _ = self.cartesian_coords(theta)
                dist = np.sqrt((x - xn) ** 2 + (y - yn) ** 2)

                if dist >= self.tolerance:
                    return theta
        return thetai

    def select_next_collision_complex(self, x, y, theta_current, solutions):
        """Select next collision point when there are more than 2 valid solutions."""
        # Filter out invalid solutions and current position
        valid_solutions = []
        for sol in solutions:
            if sol != 0.0 and abs(theta_current - sol) > self.tolerance:
                valid_solutions.append(sol)

        if not valid_solutions:
            return 0.0

        # Check which solutions represent valid trajectories
        final_solutions = []

        for theta_sol in valid_solutions:
            xn, yn, _ = self.cartesian_coords(theta_sol)
            direction_x = xn - x
            direction_y = yn - y

            # Check if trajectory stays inside boundary
            valid_trajectory = True

            for j in range(self.trajectory_steps + 1):
                t_param = j / self.trajectory_steps
                x_test = x + t_param * direction_x
                y_test = y + t_param * direction_y

                theta_test = np.arctan2(y_test, x_test)
                r_boundary = self.boundary_radius(theta_test)
                r_test = np.sqrt(x_test**2 + y_test**2)

                if r_test > r_boundary + self.tolerance:
                    valid_trajectory = False
                    break

            if valid_trajectory:
                final_solutions.append(theta_sol)

        # Return the first valid solution
        if final_solutions:
            return final_solutions[0]
        else:
            return 0.0

    def single_trajectory(self, theta_initial, alpha_initial, max_iterations=1000):
        """
        Simulate a single trajectory starting from given initial conditions.

        Parameters:
            - theta_initial: Initial angle on boundary
            - alpha_initial: Initial angle of incidence
            - max_iterations: Maximum number of collisions to simulate

        Returns:
            - A tuple containing:
                - trajectory: List of (theta, alpha) pairs representing the trajectory
                - points: Points on the boundary of the billiard table where intersections occurred, in cartesian coordinates.
        """
        points = []
        trajectory = []
        theta = theta_initial
        alpha = alpha_initial

        for i in range(max_iterations):
            # Current position on boundary
            x, y, r = self.cartesian_coords(theta)
            points.append([x, y])

            # Tangent angle at current position
            phi = self.boundary_tangent_angle(theta)

            # Find intersection points
            solutions = self.find_intersections(x, y, alpha, phi)

            # Count non-zero solutions
            non_zero_solutions = [s for s in solutions if s != 0.0]

            # Select next collision point
            if len(non_zero_solutions) <= 2:
                theta_next = self.select_next_collision_simple(theta, x, y, solutions)
            else:
                theta_next = self.select_next_collision_complex(x, y, theta, solutions)

            if theta_next == 0.0:
                break

            # Calculate new angle of incidence
            x_next, y_next, r_next = self.cartesian_coords(theta_next)
            phi_next = self.boundary_tangent_angle(theta_next)

            alpha_next = phi_next - (alpha + phi)
            alpha_next = alpha_next % np.pi

            if alpha_next < 0:
                alpha_next += np.pi

            # Store current state
            trajectory.append((theta_next, alpha_next))

            # Update for next iteration
            theta = theta_next
            alpha = alpha_next

        return (trajectory, points)

    def simulate(
        self,
        alpha_min=0.0,
        alpha_max=None,
        theta_min=0.0,
        theta_max=None,
        num_alpha=None,
        num_theta=None,
        iterations_per_trajectory=1000,
    ):
        """
        Generate phase space data and trajectory data by sampling initial conditions.

        Parameters:
        - alpha_min, alpha_max: Range for initial angle of incidence
        - theta_min, theta_max: Range for initial position angle
        - num_alpha, num_theta: Number of grid points in each direction
        - iterations_per_trajectory: Number of iterations per trajectory

        Returns:
        - A tuple containing:
            - phase_space_data: DataFrame with columns ['theta', 'alpha']
            - trajectory_data: An array of tuples containing x and y points where collisions with the boundary occurred.
        """
        if alpha_max is None:
            alpha_max = np.pi
        if theta_max is None:
            theta_max = 2 * np.pi
        if num_alpha is None:
            num_alpha = self.grid_density
        if num_theta is None:
            num_theta = self.grid_density

        phase_space_data = []

        alpha_step = (alpha_max - alpha_min) / num_alpha if num_alpha > 0 else 0
        theta_step = (theta_max - theta_min) / num_theta if num_theta > 0 else 0

        for n in range(num_alpha + 1):
            alpha_initial = alpha_min + n * alpha_step
            print(f"Processing alpha grid point {n + 1}/{num_alpha + 1}")

            for k in range(num_theta + 1):
                theta_initial = theta_min + k * theta_step

                # Simulate trajectory
                (trajectory, points) = self.single_trajectory(
                    theta_initial, alpha_initial, iterations_per_trajectory
                )

                # Add all points to phase space data
                for theta_point, alpha_point in trajectory:
                    phase_space_data.append(
                        {"theta": theta_point, "alpha": alpha_point}
                    )

        return (pd.DataFrame(phase_space_data), points)

    def plot_boundary(self, ax=None, num_points=1000):
        """Plot the boundary of the billiard."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        theta_vals = np.linspace(0, 2 * np.pi, num_points)
        r_vals = [self.boundary_radius(theta) for theta in theta_vals]
        x_vals = [r * np.cos(theta) for r, theta in zip(r_vals, theta_vals)]
        y_vals = [r * np.sin(theta) for r, theta in zip(r_vals, theta_vals)]

        ax.plot(x_vals, y_vals, "b-", linewidth=2, label="Boundary")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        return ax

    def step_map(self, theta, alpha):
        """A helper to perform exactly one collision step."""
        phi = self.boundary_tangent_angle(theta)
        x, y, _ = self.cartesian_coords(theta)
        solutions = self.find_intersections(x, y, alpha, phi)

        non_zero = [s for s in solutions if s != 0.0]
        if len(non_zero) <= 2:
            theta_next = self.select_next_collision_simple(theta, x, y, solutions)
        else:
            theta_next = self.select_next_collision_complex(x, y, theta, solutions)

        if theta_next == 0.0:
            return None

        phi_next = self.boundary_tangent_angle(theta_next)
        alpha_next = (phi_next - (alpha + phi)) % np.pi
        return theta_next, alpha_next

    def plot_trajectories(self, trajectory_data, ax=None):
        """Plot the trajectory space."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 6))
        prev_traj = None
        for next_traj in trajectory_data:
            (x, y) = next_traj
            ax.plot([x], [y], "r.")
            if prev_traj != None:
                (x1, y1) = prev_traj
                ax.plot([x, x1], [y, y1], "r-", alpha=0.4, label="Trajectory")
            prev_traj = next_traj

    def plot_phase_space(self, phase_space_data, ax=None):
        """Plot the phase space (Poincaré section)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            phase_space_data["theta"], phase_space_data["alpha"], s=0.1, alpha=0.6
        )
        ax.set_xlabel("θ (angle on boundary)")
        ax.set_ylabel("α (angle of incidence)")
        ax.set_title("Phase Space (Poincaré Section)")
        ax.grid(True, alpha=0.3)

        return ax

    def save_phase_space_data(self, phase_space_data, filename="phase_espace.dat"):
        """Save phase space data to file in format compatible with original Fortran output."""
        with open(filename, "w") as f:
            for _, row in phase_space_data.iterrows():
                f.write(f"{row['theta']:.15e} {row['alpha']:.15e}\n")
        print(f"Phase space data saved to {filename}")

    def r(self, theta):
        """Calculate the radial coordinate at angle theta."""
        return self.r0 / (1.0 + self.eccentricity * np.cos(self.m * theta))

    def rPrime(self, theta):
        """Calculate the derivative of radial coordinate with respect to theta."""
        return (
            self.r0
            * self.eccentricity
            * self.m
            * np.sin(self.m * theta)
            / (1.0 + self.eccentricity * np.cos(self.m * theta)) ** 2
        )

    def rDoublePrime(self, theta):
        """Calculate the second derivative of radial coordinate with respect to theta."""
        return (
            self.r0
            * self.eccentricity
            * self.m**2
            * (
                2 * self.eccentricity
                + np.cos(self.m * theta)
                - self.eccentricity * np.cos(self.m * theta) ** 2
            )
        ) / (1.0 + self.eccentricity * np.cos(self.m * theta)) ** 3

    def x(self, theta):
        """Calculate the x coordinate at angle theta."""
        return self.r(theta) * np.cos(theta)

    def y(self, theta):
        """Calculate the y coordinate at angle theta."""
        return self.r(theta) * np.sin(theta)

    def xPrime(self, theta):
        """Calculate the derivative of x coordinate with respect to theta."""
        return self.rPrime(theta) * np.cos(theta) - self.r(theta) * np.sin(theta)

    def yPrime(self, theta):
        """Calculate the derivative of y coordinate with respect to theta."""
        return self.rPrime(theta) * np.sin(theta) + self.r(theta) * np.cos(theta)

    def xDoublePrime(self, theta):
        """Calculate the second derivative of x coordinate with respect to theta."""
        return (
            self.rDoublePrime(theta) * np.cos(theta)
            - self.rPrime(theta) * np.sin(theta)
            - (self.rPrime(theta) * np.sin(theta) + self.r(theta) * np.cos(theta))
        )

    def yDoublePrime(self, theta):
        """Calculate the second derivative of y coordinate with respect to theta."""
        return (
            self.rDoublePrime(theta) * np.sin(theta)
            + self.rPrime(theta) * np.cos(theta)
            + self.rPrime(theta) * np.cos(theta)
            - self.r(theta) * np.sin(theta)
        )

    def phi(self, theta):
        """Calculate the angle of the tangent to the boundary at theta."""
        return np.arctan2(self.yPrime(theta), self.xPrime(theta))

    def phiPrime(self, theta):
        """Calculate the derivative of phi with respect to theta."""
        return (
            self.xPrime(theta) * self.yDoublePrime(theta)
            - self.yPrime(theta) * self.xDoublePrime(theta)
        ) / (self.xPrime(theta) ** 2 + self.yPrime(theta) ** 2)

    def secSquared(self, theta_n, alpha_n):
        """
        Calculate sec^2 at phi(theta) + alpha_n. This uses cosine, as sec is not a numpy function, and
        detects a very small denominator to avoid division by zero.
        """
        gamma = self.phi(theta_n) + alpha_n
        cos_val = np.cos(gamma)

        # Avoid division by zero by ensuring cos_val is never exactly 0
        # 1e-15 is roughly the limit of float64 precision
        if np.abs(cos_val) < 1e-12:
            # return 1e30
            # The following is a recommendation by Gemini that allows for a more smooth, "fluid" cap. The 10^30 cap
            # is too extreme, and acts like a numerical "shock." Instead, we use a lower cap of 10^12. If this doesn't
            # work, we can also try 10^12. This allows the Jacobian to remain large (capturing chaos) without causing
            # catastrophic precision loss during normalization.
            return 1e12

        return 1.0 / (cos_val**2)

    def beta(self, theta_n, theta_n1, alpha_n):
        """Calculate the beta angle at theta."""
        return (
            -self.yPrime(theta_n)
            + np.tan(self.phi(theta_n) + alpha_n) * self.xPrime(theta_n)
            - self.secSquared(theta_n, alpha_n)
            * self.phiPrime(theta_n)
            * (self.x(theta_n1) - self.x(theta_n))
        )

    def gamma(self, theta_n, theta_n1, alpha_n):
        """Calculate the gamma angle at theta."""
        return self.yPrime(theta_n1) - np.tan(
            alpha_n + self.phi(theta_n)
        ) * self.xPrime(theta_n1)

    def delta(self, theta_n, theta_n1, alpha_n):
        """Calculate the delta angle at theta."""
        return -self.secSquared(theta_n, alpha_n) * (self.x(theta_n1) - self.x(theta_n))

    def betaOverGamma(self, theta_n, theta_n1, alpha_n):
        denom = self.gamma(theta_n, theta_n1, alpha_n)
        # Add a safety floor to prevent division by near-zero.
        # Raised to 1e-12
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return self.beta(theta_n, theta_n1, alpha_n) / denom

    def deltaOverGamma(self, theta_n, theta_n1, alpha_n):
        denom = self.gamma(theta_n, theta_n1, alpha_n)
        # Add a same safety floor
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return self.delta(theta_n, theta_n1, alpha_n) / denom

    def jacobian(self, theta_n, theta_n1, alpha_n):
        """Calculate the Jacobian matrix at theta."""
        return np.array(
            [
                [
                    -self.betaOverGamma(theta_n, theta_n1, alpha_n),
                    -self.deltaOverGamma(theta_n, theta_n1, alpha_n),
                ],
                [
                    self.phiPrime(theta_n1)
                    * -self.betaOverGamma(theta_n, theta_n1, alpha_n)
                    - self.phiPrime(theta_n),
                    self.phiPrime(theta_n1)
                    * -self.deltaOverGamma(theta_n, theta_n1, alpha_n)
                    - 1,
                ],
            ]
        )
