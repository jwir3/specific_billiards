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


def run_convergence_task(args):
    r0, ecc, m, n_iters, t0, a0 = args
    hb = Heartbeat()
    sim = BilliardsNonTimeDependent(r0, ecc, m)
    return sim.get_convergence_data(t0, a0, n_iters, heartbeat=hb)


class BilliardsNonTimeDependent:
    """
    Python implementation of non-time-dependent billiards simulation.

    This class simulates the dynamics of a billiard ball in an elliptical boundary
    that doesn't change with time. The boundary is defined by:
    r = r0 / (1 + eccentricity * cos(m * theta))

    Parameters:
    - r0: Base radius (1.0)
    - m: Mode number (Must be an integer >= 1). This controls the rotational symmetry of the billiard shape, and can be
         thought of as the number of "lobes" the oval-like billiard will have. Defaults to 3.
    - eccentricity: Eccentricity parameter (0.1)
    - tolerance: Numerical tolerance (1e-12)
    - passes: Number of steps for root finding (1000)
    - trajectory_steps: Number of steps for trajectory checking (200)
    - bisection_iterations: Maximum iterations for bisection method (1000000)
    - grid_density: Number of grid points for phase space sampling (10)
    - save_density: Number of save points for trajectory plotting (2000)
    """

    def __init__(self, r0=1.0, eccentricity=0.1, mode=3):
        # Parameters from Fortran module, translated from Portuguese
        self.r0 = r0
        self.m = mode
        self.eccentricity = eccentricity
        self.tolerance = 1e-12
        self.root_passes = 100  # Set to 100 to reduce the computation time 4/14/26
        self.trajectory_steps = 200
        self.bisection_iterations = 1000000
        self.grid_density = 10

        # We need about 2000 save points for a smooth plot.
        self.save_density = 2000

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

    def get_convergence_data(self, theta0, alpha0, n_iters=2, heartbeat=None):
        """Returns (n_values, lambda_values, error_vals) using stable running stats."""
        curr_theta, curr_alpha = theta0, alpha0
        v = np.array([1.0, 0.0])

        # Stable running statistics (Welford's algorithm)
        running_mu = 0.0
        running_m2 = 0.0  # Sum of squares of differences from the mean

        n_vals = []
        lambda_vals = []
        error_vals = []

        report_interval = max(1, n_iters // 100)
        save_interval = max(1, n_iters // self.save_density)

        for i in range(1, n_iters + 1):
            base = self.step_map(curr_theta, curr_alpha)
            if not base:
                break

            (theta_next, alpha_next) = base
            J = self.jacobian(curr_theta, theta_next, curr_alpha)

            v = J @ v
            norm = np.linalg.norm(v)
            x = np.log(norm)  # The local exponent

            # --- Stable Running Mean & Variance Update ---
            delta_mean = x - running_mu
            running_mu += delta_mean / i
            delta2 = x - running_mu
            running_m2 += delta_mean * delta2

            v /= norm
            curr_theta, curr_alpha = base

            # Subsampling Logic
            if i <= 1000 or i % save_interval == 0 or i == n_iters:
                n_vals.append(i)
                lambda_vals.append(running_mu)

                if i > 1:
                    # Variance = M2 / (i - 1)
                    # SEM = sqrt(Variance / i)
                    variance = running_m2 / (i - 1)
                    error = np.sqrt(variance / i)
                else:
                    error = 0.0
                error_vals.append(error)

            if heartbeat and i % report_interval == 0:
                heartbeat.report(i // report_interval)

        if heartbeat:
            heartbeat.cleanup()

        return np.array(n_vals), np.array(lambda_vals), np.array(error_vals)

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
        if np.abs(cos_val) < 1e-15:
            return 1e30

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
        """Calculate the beta/gamma ratio at theta."""
        return self.beta(theta_n, theta_n1, alpha_n) / self.gamma(
            theta_n, theta_n1, alpha_n
        )

    def deltaOverGamma(self, theta_n, theta_n1, alpha_n):
        """Calculate the delta/gamma ratio at theta."""
        return self.delta(theta_n, theta_n1, alpha_n) / self.gamma(
            theta_n, theta_n1, alpha_n
        )

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
