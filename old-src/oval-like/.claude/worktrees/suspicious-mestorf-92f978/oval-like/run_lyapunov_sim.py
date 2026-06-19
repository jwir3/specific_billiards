# run_sim.py
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm  # Standard tqdm for terminal

from billiard_engine import run_convergence_task
from telemetry import Monitor


def main():
    # --- Configuration ---
    scale = 1.0
    m = 3
    epsilon = 0.1
    n_iters = int(1e3)
    output_file = "billiard_results.npz"

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

    task_args = [(scale, epsilon, m, n_iters, t, a) for t, a in seeds]

    # Setup Telemetry
    monitor = Monitor()
    monitor.clear()

    print(f"Starting simulation: m={m}, eps={epsilon}, n={n_iters:e}")
    print(f"Results will be saved to: {output_file}")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_convergence_task, arg) for arg in task_args]

        # Terminal progress bar
        pbar = tqdm(total=len(seeds) * 100, desc="Overall Progress", unit="tick")

        last_val = 0
        while any(f.running() for f in futures):
            current_val = monitor.get_total_progress()
            delta = current_val - last_val
            if delta > 0:
                pbar.update(delta)
                last_val = current_val
            time.sleep(2)  # Friendly to remote filesystems

        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        # Collect results
        results = [f.result() for f in futures]

    # --- Save Data ---
    # We save a dictionary where keys are 'seed_0', 'seed_1', etc.
    # Each contains the (n_vals, lam_vals) array
    save_dict = {f"seed_{i}": res for i, res in enumerate(results)}
    save_dict["metadata"] = np.array([m, epsilon, n_iters])  # Store params
    save_dict["seeds"] = np.array(seeds)

    np.savez_compressed(output_file, **save_dict)
    print(f"Successfully saved results for {len(seeds)} seeds.")
    monitor.clear()


if __name__ == "__main__":
    main()
