import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_c1_score_distribution(df_c1: pd.DataFrame, output_dir: Path):
    """Fig 1: Same-model score distribution across reruns."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_c1, x="case_id", y="parsed_score", hue="model")
    plt.xticks(rotation=90)
    plt.title("C1: Score Distribution across Reruns (Within-Model)")
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_c1_score_distribution.png")
    plt.close()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_dir: Path):
    """Fig 3: Score correlation heatmap (C2)."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("C2: Pairwise Score Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_score_correlation_heatmap.png")
    plt.close()

def plot_verdict_entropy(df_c2_metrics: pd.DataFrame, output_dir: Path):
    """Fig 4: Verdict entropy per case, sorted."""
    plt.figure(figsize=(10, 6))
    df_sorted = df_c2_metrics.sort_values("verdict_entropy", ascending=False)
    sns.barplot(data=df_sorted, x="case_id", y="verdict_entropy", color="skyblue")
    plt.xticks(rotation=90)
    plt.title("C2: Verdict Entropy per Case")
    plt.ylabel("Entropy (bits)")
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_verdict_entropy.png")
    plt.close()

def plot_c1_vs_c2_variability(df_grid: pd.DataFrame, output_dir: Path):
    """Fig 5: C1 outcome-level variability vs. C2 MPAD, side-by-side."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_grid, x="score_sd", y="score_mpad", hue="pattern", s=100)
    plt.title("C1 Score SD vs C2 Score MPAD")
    plt.xlabel("C1: Score Standard Deviation")
    plt.ylabel("C2: Mean Pairwise Absolute Deviation (MPAD)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_c1_vs_c2_variability.png")
    plt.close()

def plot_pairwise_agreement_heatmap(agreement_matrix: pd.DataFrame, output_dir: Path):
    """Fig 2: Pairwise model agreement heatmap (C2)."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("C2: Pairwise Verdict Agreement (Fraction Match)")
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_verdict_agreement_heatmap.png")
    plt.close()
