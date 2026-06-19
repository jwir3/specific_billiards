import argparse
import multiprocessing as mp  # Add this import
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# If you don't have tqdm installed, you can remove it or install via: pip install tqdm
from tqdm import tqdm

from billiard_engine import run_convergence_task
from telemetry import Monitor


def main():
    # Setup Telemetry
    monitor = Monitor()
    monitor.clear()

    parser = argparse.ArgumentParser(
        description="Run Epsilon Sweep for Lyapunov Exponents with IC Averaging"
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Number of iterations per epsilon"
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of epsilon steps")
    parser.add_argument(
        "--min", type=float, default=1e-3, help="Start of epsilon sweep"
    )
    parser.add_argument("--max", type=float, default=0.5, help="End of epsilon sweep")
    parser.add_argument(
        "--output", type=str, default="epsilon_sweep.csv", help="Output filename"
    )
    parser.add_argument(
        "--mode", type=int, default=3, help="Mode number (m) for the billiard shape"
    )
    parser.add_argument(
        "--extra_ics",
        type=int,
        default=0,
        help="Number of additional initial conditions to generate and average (Diego's method)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output, used for debugging"
    )
    args = parser.parse_args()

    ### CONFIGURATION
    scale = 1.0
    m = args.mode
    theta_0, alpha_0 = np.pi + 0.002, np.pi / 3.0

    # Initialize ICs array
    ics = np.empty(args.steps, dtype=object)
    ics[:] = [(theta_0, alpha_0)] * args.steps

    # Keeping your custom overrides for specific indices
    ics[0] = (theta_0, alpha_0 + 0.04)
    ics[args.steps - 1] = (3.0 * np.pi / 2.0, 3.0 * np.pi / 4.0)

    # Generate the epsilon sweep range
    epsilons = np.linspace(args.min, args.max, args.steps)

    # Prepare arguments for the process pool
    # We now include args.extra_ics as the 7th parameter in the tuple
    task_args = []
    for i in range(args.steps):
        task_args.append(
            (
                scale,
                epsilons[i],
                m,
                args.iters,
                ics[i][0],
                ics[i][1],
                args.extra_ics,
                monitor.queue,
                args.verbose,
            )
        )

    print(f"Starting sweep: m={m}, steps={args.steps}, n={args.iters:e}", flush=True)
    if args.extra_ics > 0:
        print(f"Averaging over {args.extra_ics + 1} initial conditions per step.")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_convergence_task, arg) for arg in task_args]

        # Total = steps * (extra_ics + 1) * (iterations / 500)
        ticks_per_ic = args.iters // 500
        total_ticks = args.steps * (args.extra_ics + 1) * ticks_per_ic
        pbar = tqdm(total=total_ticks, desc="Sweep Progress", unit="tick")

        while any(f.running() for f in futures):
            monitor.drain_into_pbar(pbar)  # This internally calls pbar.update()
            time.sleep(0.5)

        # Final drain to catch last-second signals
        monitor.drain_into_pbar(pbar)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        # Collect results
        results = []
        for i, f in enumerate(futures):
            # n_vals, lambda_vals, error_vals
            # With the modified engine, these are the averaged results
            _, lambda_vals, error_vals = f.result()

            results.append(
                {
                    "epsilon": epsilons[i],
                    "lyapunov": lambda_vals[-1],
                    "error": error_vals[-1],
                    "m": m,
                    "iterations": args.iters,
                    "extra_ics_count": args.extra_ics,
                }
            )

    # --- Save Data ---
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print(f"\nResults (including metadata) saved to {args.output}", flush=True)
    monitor.clear()


if __name__ == "__main__":
    main()
