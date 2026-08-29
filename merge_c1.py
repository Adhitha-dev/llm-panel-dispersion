import pandas as pd
import glob

# Find all C1 results.csv
c1_files = glob.glob("experiments/EXP_*_C1/tables/results.csv")
print("Found C1 files:", c1_files)

dfs = [pd.read_csv(f) for f in c1_files]
master_df = pd.concat(dfs, ignore_index=True)
master_df.to_csv("experiments/master_c1_results.csv", index=False)
print("Saved master_c1_results.csv")
