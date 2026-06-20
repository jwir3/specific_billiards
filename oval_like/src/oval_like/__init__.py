import os
from collections.abc import Iterator
from typing import TypedDict, Unpack

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
        super().__init__(num_extra_ics)

        self.r0 = r0
        self.m = mode
        self.eccentricity = eccentricity
        self.tolerance = 1e-12
        self.root_passes = 100
        self.trajectory_steps = 200
        self.bisection_iterations = 1000000
        self.grid_density = 10
        self.save_density = 2000
        self.should_output_logs = should_output_logs

        if self.should_output_logs and not os.path.exists("logs"):
            os.makedirs("logs")

    class ResolverKwargs(TypedDict, total=False):
        """Keyword arguments expected by the boundary resolver.

        Attributes:
            theta (float | str | int): The angle parameter used to calculate the boundary.
        """

        theta: float | str | int

    def _resolve_theta(self, **kwargs: Unpack[ResolverKwargs]) -> float:
        """Resolves and validates the theta parameter from the provided keyword arguments.

        Parameters:
            **kwargs: Arbitrary keyword arguments unpacked from ResolverKwargs.
                Keyword Args:
                    theta (float | str | int): The required angle parameter.

        Returns:
            float: The explicitly casted float value of theta.

        Raises:
            BoundaryException: If 'theta' is completely missing from the provided kwargs.
        """
        if "theta" not in kwargs:
            raise BoundaryException("theta was not provided")

        # The type checker now knows kwargs["theta"] is float | str | int
        theta: float = float(kwargs["theta"])
        return theta

    @override
    def boundary_point(self, **kwargs: Unpack[ResolverKwargs]) -> float:
        """
        Calculate the boundary radius at angle theta.

        This actually calculates the radius of the boundary at some angle, theta, in polar coordinates.

        Parameters:
            **kwargs: Arbitrary keyword arguments. Note: This is more specific than the base class' architecture.
              - theta (float): The angle, in polar coordinates, at which to calculate the radius.

        Returns:
            The radius of the billiard at this point.
        """

        # The double-asterisk unpacks these into a dictionary.
        theta = self._resolve_theta(**kwargs)

        return float(self.r0 / (1.0 + self.eccentricity * np.cos(self.m * theta)))

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

    @override
    def boundary_derivative(self, **kwargs: Unpack[ResolverKwargs]) -> float:
        """Calculates the derivative of the boundary radius with respect to theta.

        This method computes the rate of change of the boundary's radius at a
        specific angle by evaluating the derivative of the billiard's polar equation.

        Parameters:
            **kwargs: Arbitrary keyword arguments unpacked from ResolverKwargs.
              - theta (float | str | int): The required angle parameter passed within kwargs.

        Returns:
            float: The calculated derivative of the radius at the given angle.
        """
        theta = self._resolve_theta(**kwargs)

        derivative: float = float(
            self.r0 * self.eccentricity * self.m * np.sin(self.m * theta)
        ) / ((1.0 + self.eccentricity * np.cos(self.m * theta)) ** 2)

        return float(derivative)

    def boundary_point_cartesian(self, theta: float) -> tuple[float, float, float]:
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
        r: float = self.boundary_point(theta=theta)
        x: float = r * np.cos(theta)
        y: float = r * np.sin(theta)
        return (x, y, r)
