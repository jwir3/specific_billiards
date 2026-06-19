import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from billiard_engine import run_history_task
from telemetry import Monitor


def main():
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run Lyapunov convergence simulations for oval-like billiards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Number of collisions per trajectory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/billiard_results.npz",
        help="Output file path",
    )
    parser.add_argument(
        "--extra_ics",
        type=int,
        default=0,
        help="Number of additional ICs to generate per seed",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=3,
        help="Boundary mode 'm' (p value/number of lobes)",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Base radius r0")
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Eccentricity parameter"
    )
    parser.add_argument(
        "--logs", action="store_true", help="Enable per-IC convergence logging"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initial seeds (Theta, Alpha)
    seeds = [
        (np.pi / 8, np.pi / 8),
        (np.pi / 8, np.pi / 2),
        (np.pi / 8, 7 * np.pi / 8),
        (np.pi, np.pi / 8),
        (np.pi * 0.99, np.pi * 0.99 / 2),
        (3 * np.pi / 2, np.pi / 8),
        (3 * np.pi / 2, np.pi * 0.99 / 2),
        (np.pi, 7 * np.pi / 8),
        (3 * np.pi / 4, np.pi / 4),
        (7 * np.pi / 8, 7 * np.pi / 8),
    ]

    # Setup Telemetry
    monitor = Monitor()

    # Construct task arguments to match billiard_engine.py signature:
    # (r0, ecc, m, n_iters, t0, a0, num_extra_ics, progress_queue, should_output_logs)
    task_args = [
        (
            args.scale,
            args.epsilon,
            args.mode,
            args.iters,
            t,
            a,
            args.extra_ics,
            monitor.queue,
            args.logs,
        )
        for t, a in seeds
    ]

    # Calculate Total Ticks for the progress bar:
    # (Number of seeds) * (Trajectories per seed) * (Heartbeats per trajectory)
    # Note: Engine reports 1 tick every 500 iterations.
    total_trajectories = args.extra_ics + 1
    ticks_per_traj = args.iters // 500
    total_expected_ticks = len(seeds) * total_trajectories * ticks_per_traj

    print(f"Starting simulation: m={args.mode}, eps={args.epsilon}, n={args.iters:e}")
    print(f"Target: {len(seeds)} seeds | Output: {args.output}")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_history_task, arg) for arg in task_args]

        pbar = tqdm(total=total_expected_ticks, desc="Overall Progress", unit="tick")

        # UI Loop: Drains the queue in real-time
        while any(f.running() for f in futures):
            monitor.drain_into_pbar(pbar)
            time.sleep(0.5)

        # Final cleanup
        monitor.drain_into_pbar(pbar)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        # Collect results (n_iters, lambda, error) from each future
        results = [f.result() for f in futures]

    # --- Save Data ---
    # Since res is (history_n, history_lam, final_mu, final_sem),
    # we convert it to an object array or a list so savez can handle the inhomogeneity.
    save_dict = {}
    for i, res in enumerate(results):
        # res is a tuple: (n_history_arr, lam_history_arr, mu_scalar, sem_scalar)
        # We wrap it in np.array(..., dtype=object) to tell NumPy
        # not to try and "flatten" the inhomogeneous shapes.
        save_dict[f"seed_{i}"] = np.array(res, dtype=object)

    save_dict["metadata"] = np.array([args.mode, args.epsilon, args.iters])
    save_dict["seeds"] = np.array(seeds)

    np.savez_compressed(args.output, **save_dict)
    print(f"Successfully saved results to {args.output}")


if __name__ == "__main__":
    main()
