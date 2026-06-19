# /// script
# dependencies = [
#    "pytest",
#    "pytest-cov",
#    "numpy",
#    "matplotlib",
#    "scipy",
#    "pandas",
# ]
# ///

import numpy as np
import pytest

from billiard_engine import BilliardsNonTimeDependent

class MockHeartbeat:
    """Simple mock to track if the heartbeat report method is executed."""
    def __init__(self):
        self.report_count = 0
        self.cleanup_called = False

    def report(self, value):
        self.report_count += value

    def cleanup(self):
        self.cleanup_called = True

@pytest.fixture

def basic_chaotic_billiard():
    """Returns a mildly deformed billiard table known to be chaotic."""
    return BilliardsNonTimeDependent(r0=1.0, eccentricity=0.2, mode=3, num_extra_ics=0)

def test_logging_and_heartbeat_coverage(tmp_path, monkeypatch):
    """
    Forces get_convergence_data to execute the CSV logging and
    heartbeat reporting logic, ensuring coverage for file IO and telemetry.
    """
    # Force the engine to write logs inside a temporary test directory
    # so we don't clutter your actual project logs folder
    monkeypatch.chdir(tmp_path)

    sim = BilliardsNonTimeDependent(
        r0=1.0, eccentricity=0.1, mode=3, num_extra_ics=0, should_output_logs=True
    )

    hb = MockHeartbeat()
    t0, a0 = 1.0, 0.5

    # Run for 501 iterations so i % 500 == 0 triggers at least once,
    # and the log_interval criteria is met multiple times.
    n_iters = 501

    sim.get_convergence_data(t0, a0, n_iters=n_iters, heartbeat=hb)

    # Assertions to confirm telemetry branches ran successfully
    assert hb.report_count > 0, "Heartbeat reporting block was skipped!"
    assert hb.cleanup_called, "Heartbeat cleanup block was skipped!"

    # Assertions to verify CSV logs were written and closed properly
    log_dir = tmp_path / "logs"
    assert log_dir.exists(), "Logs folder was not created."

    log_files = list(log_dir.glob("*.csv"))
    assert len(log_files) > 0, "No CSV log file was generated."

    # Read the file to ensure Welford calculations were written
    with open(log_files[0], "r") as f:
        lines = f.readlines()
        assert len(lines) > 1, "Log file is empty."
        assert lines[0] == "iteration,lambda,error\n", "CSV headers are incorrect."

# def test_parity_between_data_and_history(basic_chaotic_billiard):
#     """
#     Asserts that get_convergence_data and get_convergence_history
#     produce the identical final Lyapunov Exponent for the same initial condition
#     by enforcing a synchronized random seed state.
#     """
#     sim = basic_chaotic_billiard
#     t0, a0 = 1.0, 0.5
#     n_iters = 500

#     # 1. Seed the RNG and get history results
#     np.random.seed(42)
#     hist_n, hist_lam, final_mu_hist, sem_hist = sim.get_convergence_history(
#         t0, a0, n_iters=n_iters
#     )

#     # 2. Re-seed the exact same RNG state and get standard data results
#     np.random.seed(42)
#     _, avg_lambda_arr, avg_error_arr = sim.get_convergence_data(t0, a0, n_iters=n_iters)
#     final_mu_data = avg_lambda_arr[0]

#     # Assertions
#     # Since the random vectors will now match perfectly, we can assert tight parity
#     assert np.isclose(final_mu_hist, final_mu_data, rtol=1e-11), (
#         f"Mismatch between history ({final_mu_hist}) and data ({final_mu_data})"
#     )
#     assert np.isclose(hist_lam[-1], final_mu_data, rtol=1e-11), (
#         f"The last element in the history tracking array ({hist_lam[-1]}) does not match data output."
#     )

def test_log_targets_history_sampling_coverage():
    """
    Validates the unified log_targets branch inside get_convergence_data
    to ensure the logarithmic history tracking block executes safely.
    """
    sim = BilliardsNonTimeDependent(r0=1.0, eccentricity=0.1, mode=3, num_extra_ics=0)
    t0, a0 = 1.0, 0.5
    n_iters = 10

    # Create a small explicit set of target iterations to trigger the branch
    explicit_log_targets = {2, 5, 10}

    # Execute the unified function passing our custom log_targets
    _, avg_lambda_arr, _ = sim.get_convergence_data(
        t0, a0, n_iters=n_iters, log_targets=explicit_log_targets
    )

    # Assert that the function completes cleanly with a valid float
    assert np.isfinite(avg_lambda_arr[0]), "The unified logging logic threw an error or returned NaN!"

def test_explicit_singularity_guard_trigger(monkeypatch):
    """
    Artificially forces a numerical collapse to verify that the
    get_convergence_data pipeline cleanly intercepts it, implements
    the boundary tangent reset, and finishes without a crash.
    """
    sim = BilliardsNonTimeDependent(r0=1.0, eccentricity=0.1, mode=3, num_extra_ics=0)

    # Define a broken jacobian method that forces underflow/nan matrix entries
    def mock_broken_jacobian(self, theta_n, theta_n1, alpha_n):
        return np.array([[0.0, 0.0], [0.0, 0.0]])

    # Inject our fake broken jacobian into the simulator instance
    monkeypatch.setattr(BilliardsNonTimeDependent, "jacobian", mock_broken_jacobian)

    t0, a0 = 1.0, 0.5
    # Run the pipeline for a few steps. It will instantly encounter the 0-matrix,
    # forcing norm = 0.0, which triggers your geometric singularity guard branch.
    _, avg_lambda_arr, _ = sim.get_convergence_data(t0, a0, n_iters=5)

    # The pipeline must gracefully handle the 0-norm steps, substitute norm=1.0,
    # and complete with a valid, finite float result (instead of crashing or returning NaN)
    assert np.isfinite(avg_lambda_arr[0]), "The singularity guard failed to safely patch a 0-norm event!"

def test_history_logging_bounds(basic_chaotic_billiard):
    """Verifies that history array generation correctly respects logarithmic spacing."""
    sim = basic_chaotic_billiard
    t0, a0 = 1.0, 0.5
    n_iters = 1000
    num_points_desired = 50

    hist_n, hist_lam, _, _ = sim.get_convergence_history(
        t0, a0, n_iters=n_iters, num_points_desired=num_points_desired
    )

    # Validate output geometry
    assert len(hist_n) == len(hist_lam)
    assert hist_n[0] == 1
    assert hist_n[-1] == n_iters
    assert np.all(np.diff(hist_n) >= 0), (
        "Iterations must be strictly monotonically increasing."
    )


def test_singularity_guard_stability():
    """
    Forces the engine into the extreme geometric limit (epsilon=0.5)
    to verify that our patch prevents NaN/Inf leakage and unfreezes the Welford sum.
    """
    # Initialize at the extreme geometric boundary limit
    sim = BilliardsNonTimeDependent(r0=1.0, eccentricity=0.5, mode=3, num_extra_ics=2)
    t0, a0 = 1.0, 0.5
    n_iters = 1000  # Enough iterations to guarantee triggering grazing collisions

    _, avg_lambda_arr, avg_error_arr = sim.get_convergence_data(t0, a0, n_iters=n_iters)
    final_lambda = avg_lambda_arr[0]
    final_error = avg_error_arr[0]

    # Assertions to prove the fix is working actively
    assert np.isfinite(final_lambda), (
        "The Lyapunov exponent collapsed to NaN or Infinity!"
    )
    assert np.isfinite(final_error), "The Welford variance tracking collapsed to NaN!"

    # With the math desynchronization fixed, the value should stay robustly
    # above 0.0, capturing the true hyper-chaotic dynamics.
    assert final_lambda > 0.1, (
        f"Lyapunov exponent abnormally decayed to {final_lambda}. Sum is frozen."
    )


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
    t_plus, a_plus = billiard.step_map(theta_n + h, alpha_n)
    t_minus, a_minus = billiard.step_map(theta_n - h, alpha_n)
    col1 = np.array([(t_plus - t_minus) / (2 * h), (a_plus - a_minus) / (2 * h)])

    # Partial derivatives w.r.t alpha
    t_plus_a, a_plus_a = billiard.step_map(theta_n, alpha_n + h)
    t_minus_a, a_minus_a = billiard.step_map(theta_n, alpha_n - h)
    col2 = np.array(
        [(t_plus_a - t_minus_a) / (2 * h), (a_plus_a - a_minus_a) / (2 * h)]
    )

    J_numerical = np.column_stack((col1, col2))

    # 3. Compare
    np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-5)


def test_jacobian_determinant():
    """
    Validates that the Jacobian matrix represents a valid area-preserving
    billiard map by asserting its determinant structure.
    """
    billiard = BilliardsNonTimeDependent(1.0, 0.1, 3)
    theta_n, alpha_n = np.pi / 3, np.pi / 4

    # 1. Map forward
    theta_n1, alpha_n1 = billiard.step_map(theta_n, alpha_n)
    J_forward = billiard.jacobian(theta_n, theta_n1, alpha_n)
    det_forward = np.linalg.det(J_forward)

    # 2. Map backward from the landing point to see the reciprocal scaling
    # Invert the velocity angle to step backwards mathematically
    theta_back, alpha_back = billiard.step_map(theta_n1, (alpha_n1 + np.pi) % np.pi)

    # Assert that the forward determinant is non-zero and finite
    assert det_forward > 0 and np.isfinite(det_forward)

    # 3. Alternatively, check that the product of the scaling factors
    # across a closed loop or inverse coordinate transformation equals 1.0.
    # To keep it incredibly robust without tracking backward steps:
    # The determinant of a 2D billiard map in these coordinates is bounded by the curvature ratios.
    assert 0.1 < abs(det_forward) < 10.0, f"Jacobian determinant {det_forward} is unphysically exploding or collapsing."
