import numpy as np
import pytest

# from argus_engine.util import MockHeartbeat
from oval_like import OvalLikeBilliard


@pytest.fixture
def basic_chaotic_billiard() -> OvalLikeBilliard:
    """Returns a mildly deformed billiard table known to be chaotic."""
    return OvalLikeBilliard(r0=1.0, eccentricity=0.2, mode=3, num_extra_ics=0)


def test_basic_billiard_construction(basic_chaotic_billiard: OvalLikeBilliard):
    billiard = basic_chaotic_billiard
    assert billiard is not None


def test_boundary_returns_correct_number_of_points(
    basic_chaotic_billiard: OvalLikeBilliard,
):
    """Ensure the boundary generator yields the exact number of requested points."""
    num_points = 500
    points_iterator = basic_chaotic_billiard.boundary(num_points=num_points)

    # Convert iterator to list to count and evaluate
    points = list(points_iterator)

    assert len(points) == num_points


def test_boundary_yields_valid_cartesian_coordinates(
    basic_chaotic_billiard: OvalLikeBilliard,
):
    """Verify that the boundary method converts polar to Cartesian properly."""
    points_iterator = basic_chaotic_billiard.boundary(num_points=10)

    for x, y in points_iterator:
        # Check that types are strictly floats, not arrays or Any
        assert isinstance(x, float) or isinstance(x, np.float64)
        assert isinstance(y, float) or isinstance(y, np.float64)
        # Ensure no NaNs or Infs leaked through the linspace math
        assert np.isfinite(x)
        assert np.isfinite(y)


# --- Tests for boundary_derivative() ---


def test_boundary_derivative_at_zero(basic_chaotic_billiard: OvalLikeBilliard):
    """
    Test derivative at theta = 0.
    Because sin(m * 0) is 0, the entire numerator should evaluate to 0.0.
    """
    derivative = basic_chaotic_billiard.boundary_derivative(theta=0.0)

    assert derivative == pytest.approx(0.0)


def test_boundary_derivative_at_peak(basic_chaotic_billiard: OvalLikeBilliard):
    """
    Test derivative at a known peak.
    With mode=3, setting theta = pi/6 makes (mode * theta) = pi/2.
    sin(pi/2) = 1, cos(pi/2) = 0.
    Equation: (1.0 * 0.2 * 3 * 1) / (1 + 0)^2 = 0.6.
    """
    # Peak occurs where mode * theta = pi/2
    theta_peak = np.pi / 6
    derivative = basic_chaotic_billiard.boundary_derivative(theta=theta_peak)

    assert derivative == pytest.approx(0.6)


def test_boundary_derivative_accepts_kwargs_unpacking(
    basic_chaotic_billiard: OvalLikeBilliard,
):
    """Verify that passing dictionary unpacking works as intended by the TypedDict."""
    kwargs = {"theta": np.pi / 6}
    derivative = basic_chaotic_billiard.boundary_derivative(**kwargs)

    assert derivative == pytest.approx(0.6)


# def test_logging_and_heartbeat_coverage(tmp_path, monkeypatch):
#     """
#     Forces get_convergence_data to execute the CSV logging and
#     heartbeat reporting logic, ensuring coverage for file IO and telemetry.
#     """
#     # Force the engine to write logs inside a temporary test directory
#     # so we don't clutter your actual project logs folder
#     monkeypatch.chdir(tmp_path)

#     sim = OvalLikeBilliard(
#         r0=1.0, eccentricity=0.1, mode=3, num_extra_ics=0, should_output_logs=True
#     )

#     hb = MockHeartbeat()
#     t0, a0 = 1.0, 0.5

#     # Run for 501 iterations so i % 500 == 0 triggers at least once,
#     # and the log_interval criteria is met multiple times.
#     n_iters = 501

#     # sim.get_convergence_data(t0, a0, n_iters=n_iters, heartbeat=hb)

#     # Assertions to confirm telemetry branches ran successfully
#     assert hb.report_count > 0, "Heartbeat reporting block was skipped!"
#     assert hb.cleanup_called, "Heartbeat cleanup block was skipped!"

#     # Assertions to verify CSV logs were written and closed properly
#     log_dir = tmp_path / "logs"
#     assert log_dir.exists(), "Logs folder was not created."

#     log_files = list(log_dir.glob("*.csv"))
#     assert len(log_files) > 0, "No CSV log file was generated."

#     # Read the file to ensure Welford calculations were written
#     with open(log_files[0], "r") as f:
#         lines = f.readlines()
#         assert len(lines) > 1, "Log file is empty."
#         assert lines[0] == "iteration,lambda,error\n", "CSV headers are incorrect."


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


# def test_log_targets_history_sampling_coverage():
#     """
#     Validates the unified log_targets branch inside get_convergence_data
#     to ensure the logarithmic history tracking block executes safely.
#     """
#     sim = BilliardsNonTimeDependent(r0=1.0, eccentricity=0.1, mode=3, num_extra_ics=0)
#     t0, a0 = 1.0, 0.5
#     n_iters = 10

#     # Create a small explicit set of target iterations to trigger the branch
#     explicit_log_targets = {2, 5, 10}

#     # Execute the unified function passing our custom log_targets
#     _, avg_lambda_arr, _ = sim.get_convergence_data(
#         t0, a0, n_iters=n_iters, log_targets=explicit_log_targets
#     )

#     # Assert that the function completes cleanly with a valid float
#     assert np.isfinite(avg_lambda_arr[0]), (
#         "The unified logging logic threw an error or returned NaN!"
#     )
