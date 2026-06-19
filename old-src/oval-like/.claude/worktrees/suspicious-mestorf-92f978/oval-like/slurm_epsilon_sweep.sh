#!/bin/bash
#SBATCH --job-name=billiard_sweep
#SBATCH --array=0-31
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=runs/latest_run_logs/slurm-%A_%a.out
#
export PATH="$HOME/.local/bin:$PATH"
cd $SLURM_SUBMIT_DIR

# 3. Run the Epsilon Sweep
uv run python run_epsilon_sweep_hpc.py \
    --steps 32 \
    --iters 10000000 \
    --mode 3 \
    --min 0.001 \
    --max 0.05 \
    --outdir "$RUN_DIR/data" > "$RUN_DIR/logs/task_$SLURM_ARRAY_TASK_ID.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID complete."
