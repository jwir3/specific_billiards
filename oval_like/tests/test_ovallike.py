import pytest

# from argus_engine.util import MockHeartbeat
from oval_like import OvalLikeBilliard


@pytest.fixture
def basic_chaotic_billiard():
    """Returns a mildly deformed billiard table known to be chaotic."""
    return OvalLikeBilliard(r0=1.0, eccentricity=0.2, mode=3, num_extra_ics=0)


def test_basic_billiard_construction(basic_chaotic_billiard):
    billiard = basic_chaotic_billiard
    assert billiard is not None


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
