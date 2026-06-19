import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    pattern = os.path.join(args.outdir, "epsilon_sweep_results_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found in {args.outdir}")
        return

    print(f"Merging {len(files)} files...")
    df_list = [pd.read_csv(f) for f in files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Sort by epsilon so the final graph looks correct
    merged_df = merged_df.sort_values("epsilon")

    # Extract the timestamp from the directory name (e.g., 'run_20260502_184000')
    # This assumes the outdir is 'runs/run_TIMESTAMP/data'
    parent_dir = os.path.basename(os.path.dirname(args.outdir))
    timestamp = parent_dir.replace("run_", "")

    output_filename = f"epsilon_sweep_results_{timestamp}.csv"
    output_path = os.path.join(args.outdir, output_filename)

    merged_df.to_csv(output_path, index=False)
    print(f"Final dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
