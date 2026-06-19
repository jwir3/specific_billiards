import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from billiard_engine import run_convergence_task
from telemetry import Monitor


def main():
    parser = argparse.ArgumentParser(
        description="Run Epsilon Sweep for Lyapunov Exponents with Metadata"
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
    # Added argument for 'm' to make the CLI even more flexible
    parser.add_argument(
        "--mode", type=int, default=3, help="Mode number (m) for the billiard shape"
    )
    args = parser.parse_args()

    ### CONFIGURATION
    scale = 1.0
    m = args.mode
    theta_0, alpha_0 = np.pi + 0.002, np.pi / 3.0

    ics = np.empty(args.steps, dtype=object)
    ics[:] = [(theta_0, alpha_0)] * args.steps
    ics[0] = (theta_0, alpha_0 + 0.04)
    ics[args.steps - 1] = (3.0 * np.pi / 2.0, 3.0 * np.pi / 4.0)

    # Generate the epsilon sweep range
    epsilons = np.linspace(args.min, args.max, args.steps)

    task_args = np.empty(args.steps, dtype=object)

    for i in range(args.steps):
        task_args[i] = (scale, epsilons[i], m, args.iters, ics[i][0], ics[i][1])

    # task_args = [(scale, eps, m, args.iters, theta_0, alpha_0) for eps in epsilons]

    # Setup Telemetry
    monitor = Monitor()
    monitor.clear()

    print(f"Starting sweep: m={m}, steps={args.steps}, n={args.iters:e}", flush=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_convergence_task, arg) for arg in task_args]
        pbar = tqdm(total=len(epsilons) * 100, desc="Sweep Progress", unit="tick")

        last_val = 0
        while any(f.running() for f in futures):
            current_val = monitor.get_total_progress()
            delta = current_val - last_val
            if delta > 0:
                pbar.update(delta)
                last_val = current_val
            time.sleep(2)

        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        # Collect results
        results = []
        for i, f in enumerate(futures):
            # Now returns n, lambda, error
            _, lambda_vals, error_vals = f.result()
            results.append(
                {
                    "epsilon": epsilons[i],
                    "lyapunov": lambda_vals[-1],
                    # We only save the final error estimate
                    "error": error_vals[-1],
                    "m": m,
                    "iterations": args.iters,
                }
            )
    # --- Save Data ---
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print(f"\nResults (including metadata) saved to {args.output}", flush=True)
    monitor.clear()


if __name__ == "__main__":
    main()
