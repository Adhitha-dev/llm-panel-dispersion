import pandas as pd
from pathlib import Path

dfs = []
for p in Path("experiments").glob("EXP_*_C2_*/tables/results.csv"):
    print(f"Reading {p}")
    dfs.append(pd.read_csv(p))

combined = pd.concat(dfs, ignore_index=True)
out_dir = Path("experiments/EXP_C2_COMBINED/tables")
out_dir.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_dir / "results.csv", index=False)
print(f"Combined {len(dfs)} C2 result files. Total rows: {len(combined)}")
