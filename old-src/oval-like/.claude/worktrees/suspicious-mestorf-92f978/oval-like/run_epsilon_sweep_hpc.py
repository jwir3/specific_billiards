# This can be run with the following commands:
# # 1. Set the run directory
# export SWEEP_DIR="runs/run_$(date +"%Y%m%d_%H%M%S")"
#
# 2. Submit the Epsilon Sweep and capture the Job ID
# 'afterok' ensures merging only happens if all tasks finish successfully
# SWEEP_JOB_ID=$(sbatch --parsable --export=ALL,SWEEP_DIR=$SWEEP_DIR slurm_epsilon_sweep.sh)
#
# 3. Submit the Merge Job as a dependency
# sbatch --dependency=afterok:$SWEEP_JOB_ID --job-name=merge_billiards \
#        --ntasks=1 --mem=4G --time=00:30:00 \
#        --wrap="uv run python merge_results.py --outdir $SWEEP_DIR/data"
import argparse
import os

import numpy as np
import pandas as pd

from billiard_engine import run_convergence_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--min", type=float, default=1e-3)
    parser.add_argument("--max", type=float, default=0.5)
    parser.add_argument("--mode", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    # Instead of using a hardcoded --steps argument:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))  # Slurm provides this!

    # Generate the range based on the ACTUAL size of the Slurm array
    all_epsilons = np.linspace(args.min, args.max, num_tasks)
    epsilon = all_epsilons[task_id]

    # Ensure output directory exists (exist_ok=True handles concurrent tasks)
    os.makedirs(args.outdir, exist_ok=True)

    # Generate epsilon for this specific task
    all_epsilons = np.linspace(args.min, args.max, args.steps)
    epsilon = all_epsilons[task_id]

    # Engine execution
    scale = 1.0
    theta_0, alpha_0 = np.pi + 0.002, np.pi / 2 + 0.002

    # run_convergence_task returns (n, lambda_vals, error_vals)
    _, lambda_vals, error_vals = run_convergence_task(
        (scale, epsilon, args.mode, args.iters, theta_0, alpha_0)
    )

    # Save to the specific directory
    result = {
        "epsilon": epsilon,
        "lyapunov": lambda_vals[-1],
        "error": error_vals[-1],
        "m": args.mode,
        "iterations": args.iters,
    }

    df = pd.DataFrame([result])
    output_path = os.path.join(args.outdir, f"epsilon_sweep_results_{task_id}.csv")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
