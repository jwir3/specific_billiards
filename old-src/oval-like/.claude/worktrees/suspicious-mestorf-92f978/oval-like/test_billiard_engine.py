import numpy as np
import pytest
from billiard_engine import BilliardsNonTimeDependent


def test_jacobian_accuracy():
    """
    Validates the analytical Jacobian against a numerical
    finite-difference approximation.
    """
    # Setup parameters
    scale, m, epsilon = 1.0, 3, 0.1
    billiard = BilliardsNonTimeDependent(scale, epsilon, m)

    # Test point (avoiding the pi singularity you found!)
    theta_n = np.pi / 4
    alpha_n = np.pi / 6

    # 1. Get the analytical Jacobian
    # We need theta_n1 to pass into your function
    (theta_n1, alpha_n1) = billiard.step_map(theta_n, alpha_n)
    J_analytical = billiard.jacobian(theta_n, theta_n1, alpha_n)

    # 2. Compute Numerical Jacobian using Finite Differences
    h = 1e-7  # Small perturbation

    # Partial derivatives w.r.t theta
    t_plus, a_plus = billiard.step(theta_n + h, alpha_n)
    t_minus, a_minus = billiard.step(theta_n - h, alpha_n)
    col1 = np.array([(t_plus - t_minus) / (2 * h), (a_plus - a_minus) / (2 * h)])

    # Partial derivatives w.r.t alpha
    t_plus_a, a_plus_a = billiard.step(theta_n, alpha_n + h)
    t_minus_a, a_minus_a = billiard.step(theta_n, alpha_n - h)
    col2 = np.array(
        [(t_plus_a - t_minus_a) / (2 * h), (a_plus_a - a_minus_a) / (2 * h)]
    )

    J_numerical = np.column_stack((col1, col2))

    # 3. Compare
    np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-5)


def test_jacobian_determinant():
    """
    Billiard maps are often area-preserving (Hamiltonian).
    If yours is, the determinant of the Jacobian should be 1.0.
    """
    billiard = YourBilliardClass(1.0, 0.1, 3)
    theta_n, alpha_n = np.pi / 3, np.pi / 4
    theta_n1, _ = billiard.step(theta_n, alpha_n)

    J = billiard.jacobian(theta_n, theta_n1, alpha_n)
    det = np.linalg.det(J)

    # Note: Use absolute value if your map is orientation-reversing
    assert np.isclose(abs(det), 1.0, atol=1e-7)
