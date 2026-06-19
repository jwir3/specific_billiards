#!/bin/bash

# 1. Define the directory once
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export RUN_DIR="runs/run_$TIMESTAMP"

# 2. Create the folders
mkdir -p "$RUN_DIR/data" "$RUN_DIR/logs"
mkdir -p "runs/latest_run_logs"
ln -snf "$(basename $RUN_DIR)" runs/latest

echo "Run dir is: $RUN_DIR"

# 3. Submit the Sweep and capture the Job ID
# The --parsable flag makes it return ONLY the Job ID (e.g., 252174)
SWEEP_JOB_ID=$(sbatch --parsable --export=ALL,SWEEP_DIR=$RUN_DIR slurm_epsilon_sweep.sh)

echo "Submitted Sweep Job: $SWEEP_JOB_ID"

# 4. Submit the Collation Job from the login node
# It waits for the sweep Job ID to finish
# 4. Submit the Collation Job with ESCAPED variables
sbatch --dependency=afterok:$SWEEP_JOB_ID \
       --job-name=collate_billiards \
       --ntasks=1 --mem=4G --time=00:15:00 \
       --output="$RUN_DIR/logs/collation.log" \
       --wrap="export PATH=\"\$HOME/.local/bin:\$PATH\" && cd \$SLURM_SUBMIT_DIR && uv run python merge_results.py --outdir $RUN_DIR/data"

echo "Submitted Collation Job with dependency on $SWEEP_JOB_ID"
