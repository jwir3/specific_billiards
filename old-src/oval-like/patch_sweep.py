import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from billiard_engine import run_convergence_task
from telemetry import Monitor

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Highly optimized fine-grained parallel data patching script using ensemble core splitting."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the existing, unpolished CSV dataset containing full sweep data."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the polished, patched CSV dataset should be saved."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10**7,
        help="Number of iterations (collisions) per trajectory run. Default: 10^7."
    )
    parser.add_argument(
        "--extra_ics",
        type=int,
        default=10,
        help="Number of additional initial conditions to run per parameter configuration. Default: 10."
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        help="Space-separated list of broken eccentricity/epsilon values to isolate and compute fresh parameters for."
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Enable individual trajectory convergence logging (outputs to logs/ directory)."
    )
    return parser.parse_args()


def run_single_ic_task(ecc, ic_idx, total_ics, iters, progress_queue, should_log):
    """
    Highly optimized fine-grained task worker. Spins up an isolated engine instance
    to process exactly ONE trajectory/IC configuration. This ensures 100% core usage
    even when patching a single epsilon value.
    """
    from billiard_engine import BilliardsNonTimeDependent
    from telemetry import Heartbeat
    import os

    scale = 1.0
    m = 3

    # NOTE: These need to stay aligned with `run_epsilon_sweep.py`!
    t0 = np.pi + 0.002
    a0 = np.pi / 3.0

    hb = Heartbeat(progress_queue)

    # FIX: Pass total_ics - 1 (which matches extra_ics) to ensure the
    # ensemble list is fully populated with all randomized neighbor scouts
    sim = BilliardsNonTimeDependent(
        r0=scale, eccentricity=ecc, mode=m, num_extra_ics=total_ics - 1, should_output_logs=should_log
    )

    # Replicate Diego's method: generate the complete ensemble and isolate our target scout index safely
    all_ics = sim._generate_additional_initial_conditions(t0, a0)
    target_theta, target_alpha = all_ics[ic_idx]

    # Open process-isolated log if requested
    log_file = None
    num_points_desired = 200
    log_interval = max(1, iters // num_points_desired)

    if should_log:
        pid = os.getpid()
        log_filename = f"logs/conv_log_eps{ecc:.3f}_pid{pid}_ic{ic_idx}.csv"
        log_file = open(log_filename, "w")
        log_file.write("iteration,lambda,error\n")

    # Execute simulation for this single IC channel
    running_mu, running_m2, _, _ = sim._run_convergence_for_initial_condition(
        target_theta, target_alpha, iters, log_interval, hb, log_file, log_targets=None
    )

    if log_file:
        log_file.close()
    hb.cleanup()

    # Calculate standard sample error from Welford variance metrics
    if iters > 1:
        variance = running_m2 / (iters - 1)
        sem = np.sqrt(variance / iters)
    else:
        sem = 0.0

    return ecc, running_mu, sem


def main():
    args = parse_arguments()

    if not args.epsilons:
        print("Error: No epsilon values were given to patch. Provide values using --epsilons.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input data file '{args.input}' could not be found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    lambda_col = 'lyapunov' if 'lyapunov' in df.columns else 'lambda'
    error_col = 'error'
    eps_col = 'epsilon'
    total_steps_in_file = len(df)

    if len(args.epsilons) > total_steps_in_file:
        print(f"Error: Target epsilons count exceeds original dataset density.", file=sys.stderr)
        sys.exit(1)

    invalid_epsilons = []
    for ecc in args.epsilons:
        match_condition = np.isclose(df[eps_col], ecc, atol=1e-3)
        if not match_condition.any():
            invalid_epsilons.append(ecc)

    if invalid_epsilons:
        print(f"Error: Target epsilon(s) {invalid_epsilons} do not match dataset entries.", file=sys.stderr)
        sys.exit(1)

    print("All defensive safety guards passed successfully.")
    print("Fine-grained orchestration: Parallelizing across the trajectory ensemble level.")
    print(f"Dataset configurations: {args.input} -> {args.output}")

    monitor = Monitor()
    monitor.clear()

    # Math-scaling parameters for the total progress bar tracking
    total_trajectories = args.extra_ics + 1
    ticks_per_ic = args.iters // 500
    total_ticks = len(args.epsilons) * total_trajectories * ticks_per_ic
    pbar = tqdm(total=total_ticks, desc="Fine-Grain Patching Progress", unit="tick")

    # Dictionary to collect results grouped by epsilon: {eccentricity: ([lambdas], [errors])}
    raw_ensemble_collector = {ecc: ([], []) for ecc in args.epsilons}

    with ProcessPoolExecutor() as executor:
        futures = []
        # UNROLL THE LOOPS: Flatten out every single IC into its own standalone future task
        for ecc in args.epsilons:
            for ic_idx in range(total_trajectories):
                # FIX: Correctly call executor.submit and append the object to our futures tracking array
                f = executor.submit(
                    run_single_ic_task,
                    ecc, ic_idx, total_trajectories, args.iters, monitor.queue, args.logs
                )
                futures.append(f)

        # Active telemetry polling loop
        while any(f.running() for f in futures):
            monitor.drain_into_pbar(pbar)
            time.sleep(0.5)

        monitor.drain_into_pbar(pbar)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        # Group raw trajectory calculations back under their parental parameters
        for f in futures:
            ecc_val, ic_lambda, ic_error = f.result()
            raw_ensemble_collector[ecc_val][0].append(ic_lambda)
            raw_ensemble_collector[ecc_val][1].append(ic_error)

    # Compute final robust statistics matching the engine's standard design
    patched_results = {}
    print("\n--- Processing Fine-Grained Statistics via Median Filtering ---")
    for ecc, (lambdas, errors) in raw_ensemble_collector.items():
        avg_lambda = np.median(lambdas)
        avg_error = np.sqrt(np.sum(np.square(errors))) / len(errors)
        patched_results[ecc] = {"lambda": avg_lambda, "error": avg_error}
        print(f"Success! Epsilon: {ecc:.6f} -> Median Lambda: {avg_lambda:.6f}, SEM Error: {avg_error:.6f}")

    # Map compiled parameters back into memory dataframe
    df_patched = df.copy()
    for ecc, stats in patched_results.items():
        match_condition = np.isclose(df_patched[eps_col], ecc, atol=1e-3)
        df_patched.loc[match_condition, lambda_col] = stats["lambda"]
        df_patched.loc[match_condition, error_col] = stats["error"]
        print(f"Patched target parameter row for epsilon = {ecc} in memory.")

    df_patched.sort_values(by=eps_col, inplace=True)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_patched.to_csv(args.output, index=False)
    print(f"\nCompleted! Highly efficient patched sweep written to '{args.output}'.")
    monitor.clear()

if __name__ == "__main__":
    main()