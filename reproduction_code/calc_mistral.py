import pandas as pd
from src.metrics.calculator import compute_c1_metrics
import os
os.makedirs("experiments/EXP_20260828_153213_C1/metrics", exist_ok=True)
df_c1 = pd.read_csv("experiments/EXP_20260828_153213_C1/tables/results.csv")
c1_metrics = compute_c1_metrics(df_c1)
c1_metrics.to_csv("experiments/EXP_20260828_153213_C1/metrics/c1_metrics.csv", index=False)
print("Mistral C1 Metrics:")
print(f"Mean Flip Rate: {c1_metrics['flip_rate'].mean():.4f}")
print(f"Mean Score SD: {c1_metrics['score_sd'].mean():.4f}")
print(f"Mean Rubric Profile Distance: {c1_metrics['rubric_profile_distance'].mean():.4f}")
