import os
from typing import Iterator

import numpy as np
from argus_engine.billiard_base import Billiard, BoundaryException
from typing_extensions import override


class OvalLikeBilliard(Billiard):
    def __init__(
        self,
        r0: float = 1.0,  # the initial, or base radius
        eccentricity: float = 0.1,  # the eccentricity (how fast the lobes change)
        mode: int = 3,  # number of "lobes" of the billiard
        num_extra_ics: int = 0,  # the number of additional initial conditions to generate
        should_output_logs: bool = False,  # whether or not logs should be generated
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

    @override
    def boundary_point(self, **kwargs) -> float | tuple[float, float]:
        """
        Calculate the boundary radius at angle theta.

        This actually calculates the radius of the boundary at some angle, theta, in polar coordinates.

        Parameters:
            **kwargs: Arbitrary keyword arguments.
              - theta (float): The angle, in polar coordinates, at which to calculate the radius.

        Returns:
            The radius of the billiard at this point.
        """
        if "theta" not in kwargs or kwargs["theta"] is None:
            raise BoundaryException("theta was not provided")

        theta: float = float(kwargs["theta"])

        return self.r0 / (1.0 + self.eccentricity * np.cos(self.m * theta))

    @override
    def boundary(self, num_points: int = 1000) -> Iterator[tuple[float, float]]:
        """
        Calculate a set of points, in Cartesian Coordinates, that represent the
        boundary of the billiard.
        """

        theta_vals = np.linspace(0, 2 * np.pi, num_points)
        r_vals = [self.boundary_point(theta=theta) for theta in theta_vals]
        x_vals = [r * np.cos(theta) for r, theta in zip(r_vals, theta_vals)]
        y_vals = [r * np.sin(theta) for r, theta in zip(r_vals, theta_vals)]

        return zip(x_vals, y_vals)
