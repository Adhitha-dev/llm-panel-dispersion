import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# ── Global style ──────────────────────────────────────────────────────────────
MODEL_SHORT = {
    "models/Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "models/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "models/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}
MODEL_COLORS = {
    "Llama-3-8B": "#4C72B0",
    "Mistral-7B": "#DD8452",
    "Qwen2.5-7B": "#55A868",
}
PATTERN_COLORS = {
    "Stable task; judges converge":          "#2ecc71",
    "Distinct, stable model dispositions dominate": "#3498db",
    "Individually unstable, average converges":     "#e67e22",
    "Evaluation setting highly uncertain":          "#e74c3c",
}
sns.set_theme(style="whitegrid", font_scale=1.15)


def _shorten_models(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(lambda x: MODEL_SHORT.get(x, x))
    return df

def _shorten_matrix_labels(mat: pd.DataFrame) -> pd.DataFrame:
    mat = mat.copy()
    mat.index = [MODEL_SHORT.get(r, r) for r in mat.index]
    mat.columns = [MODEL_SHORT.get(c, c) for c in mat.columns]
    return mat


# ── Figure 1 — C1 Noise Floor: Score SD per model ─────────────────────────────
def plot_c1_score_distribution(df_c1: pd.DataFrame, output_dir: Path):
    """
    Violin + strip plot of score SD per model, with a zero-line annotation
    to clearly illustrate the noise floor.
    """
    df = _shorten_models(df_c1)

    # Compute per-case score SD per model
    sd_df = (
        df.groupby(["model", "case_id"])["parsed_score"]
        .std(ddof=1)
        .reset_index()
        .rename(columns={"parsed_score": "score_sd"})
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {m: MODEL_COLORS[m] for m in MODEL_COLORS if m in sd_df["model"].unique()}

    sns.violinplot(
        data=sd_df, x="model", y="score_sd",
        palette=palette, inner=None, cut=0, alpha=0.55, ax=ax
    )
    sns.stripplot(
        data=sd_df, x="model", y="score_sd",
        palette=palette, jitter=True, size=4, alpha=0.8, ax=ax
    )

    ax.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Zero noise floor")
    ax.set_title("C1 — Within-Model Score SD across 10 Reruns\n(Temperature = 0.0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score Standard Deviation")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_c1_noise_floor.png", dpi=150)
    plt.close()


# ── Figure 2 — C1 Flip Rate per model ─────────────────────────────────────────
def plot_c1_flip_rate(df_c1: pd.DataFrame, output_dir: Path):
    """
    Grouped bar chart: Flip Rate per model per case, sorted by mean flip rate.
    """
    df = _shorten_models(df_c1)

    def flip_rate(group):
        majority = group["parsed_verdict"].value_counts().iloc[0]
        return 1 - majority / len(group)

    flip_df = (
        df.groupby(["model", "case_id"])
        .apply(flip_rate)
        .reset_index()
        .rename(columns={0: "flip_rate"})
    )
    model_order = (
        flip_df.groupby("model")["flip_rate"].mean()
        .sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {m: MODEL_COLORS[m] for m in MODEL_COLORS}
    sns.barplot(
        data=flip_df, x="case_id", y="flip_rate", hue="model",
        hue_order=model_order, palette=palette, ax=ax
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("C1 — Verdict Flip Rate per Case\n(Lower is more stable; all models at 0 confirms noise floor)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Flip Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right", fontsize=7.5)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_c1_flip_rate.png", dpi=150)
    plt.close()


# ── Figure 3 — C2 Pairwise Verdict Agreement heatmap ─────────────────────────
def plot_pairwise_agreement_heatmap(agreement_matrix: pd.DataFrame, output_dir: Path):
    mat = _shorten_matrix_labels(agreement_matrix.set_index("model") if "model" in agreement_matrix.columns else agreement_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.eye(len(mat), dtype=bool)
    sns.heatmap(
        mat.astype(float), annot=True, fmt=".2f",
        cmap="YlGnBu", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        mask=mask, ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    # Fill diagonal separately
    for i in range(len(mat)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#f0f0f0"))
        ax.text(i + 0.5, i + 0.5, "—", ha="center", va="center", fontsize=13, color="#aaaaaa")

    ax.set_title("C2 — Pairwise Verdict Agreement\n(Fraction of cases with matching verdict)", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_verdict_agreement_heatmap.png", dpi=150)
    plt.close()


# ── Figure 4 — C2 Score Correlation heatmap ───────────────────────────────────
def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_dir: Path):
    mat = _shorten_matrix_labels(corr_matrix.set_index("model") if "model" in corr_matrix.columns else corr_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.eye(len(mat), dtype=bool)
    sns.heatmap(
        mat.astype(float), annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        mask=mask, ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    for i in range(len(mat)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#f0f0f0"))
        ax.text(i + 0.5, i + 0.5, "1.00", ha="center", va="center", fontsize=12, color="#aaaaaa")

    ax.set_title("C2 — Pairwise Score Correlation (Pearson)\nSupports Effective Viewpoints calculation", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_score_correlation_heatmap.png", dpi=150)
    plt.close()


# ── Figure 5 — C2 Verdict Entropy per case (sorted) ──────────────────────────
def plot_verdict_entropy(df_c2_metrics: pd.DataFrame, output_dir: Path):
    df_sorted = df_c2_metrics.sort_values("verdict_entropy", ascending=False).copy()

    colors = ["#e74c3c" if h > 0.5 else "#3498db" for h in df_sorted["verdict_entropy"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(df_sorted)), df_sorted["verdict_entropy"], color=colors, width=0.7, edgecolor="white")

    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["case_id"], rotation=75, ha="right", fontsize=8)
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Verdict Entropy (bits)")
    ax.set_title("C2 — Verdict Entropy per Case (Sorted)\nRed = panel split, Blue = panel agrees", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="0.5 bit threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_verdict_entropy.png", dpi=150)
    plt.close()


# ── Figure 6 — Four-Quadrant Pattern Grid ────────────────────────────────────
def plot_c1_vs_c2_variability(df_grid: pd.DataFrame, output_dir: Path):
    pattern_col = "pattern"
    palette = {k: v for k, v in PATTERN_COLORS.items() if k in df_grid[pattern_col].unique()}

    fig, ax = plt.subplots(figsize=(9, 7))

    # Compute thresholds
    x_thresh = df_grid["score_sd"].median() if df_grid["score_sd"].std() > 0 else 0.01
    y_thresh = df_grid["score_mpad"].median()

    # Draw quadrant shading
    xlim = (df_grid["score_sd"].max() * 1.15) if df_grid["score_sd"].max() > 0 else 0.1
    ylim = df_grid["score_mpad"].max() * 1.15
    ax.axvspan(0, x_thresh, ymin=0, ymax=1, color="#f9f9f9", zorder=0)
    ax.axvspan(x_thresh, xlim, ymin=0, ymax=1, color="#fff3f3", zorder=0)
    ax.axhline(y_thresh, color="#cccccc", linestyle="--", linewidth=1)
    ax.axvline(x_thresh, color="#cccccc", linestyle="--", linewidth=1)

    for pattern, group in df_grid.groupby(pattern_col):
        color = palette.get(pattern, "#aaaaaa")
        ax.scatter(group["score_sd"], group["score_mpad"],
                   label=pattern, color=color, s=90, alpha=0.85, edgecolors="white", linewidths=0.5)
        for _, row in group.iterrows():
            ax.annotate(row["case_id"], (row["score_sd"], row["score_mpad"]),
                        fontsize=6.5, alpha=0.7,
                        xytext=(4, 3), textcoords="offset points")

    # Quadrant labels
    ax.text(0.01, ylim * 0.97, "Low C1 / High C2\nDistinct model dispositions", fontsize=8, color="#3498db", va="top")
    ax.text(0.01, y_thresh * 0.45, "Low C1 / Low C2\nStable — judges converge", fontsize=8, color="#2ecc71", va="top")

    ax.set_xlim(-0.005, xlim)
    ax.set_ylim(-0.1, ylim)
    ax.set_xlabel("C1: Score Standard Deviation (Within-Model)")
    ax.set_ylabel("C2: Score MPAD (Cross-Model)")
    ax.set_title("Four-Pattern Interpretation Grid\n(C1 Within-Model Stability vs C2 Cross-Model Disagreement)", fontsize=13, fontweight="bold")
    ax.legend(title="Pattern", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_four_pattern_grid.png", dpi=150)
    plt.close()


# ── Figure 7 — C2 MPAD per case ───────────────────────────────────────────────
def plot_c2_mpad(df_c2_metrics: pd.DataFrame, output_dir: Path):
    df_sorted = df_c2_metrics.sort_values("score_mpad", ascending=False).copy()
    colors = ["#e74c3c" if m > df_c2_metrics["score_mpad"].median() else "#3498db"
              for m in df_sorted["score_mpad"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(df_sorted)), df_sorted["score_mpad"], color=colors, width=0.7, edgecolor="white")
    ax.axhline(df_c2_metrics["score_mpad"].median(), color="black", linestyle="--", linewidth=1,
               label=f"Median MPAD = {df_c2_metrics['score_mpad'].median():.2f}")
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["case_id"], rotation=75, ha="right", fontsize=8)
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Mean Pairwise Absolute Deviation")
    ax.set_title("C2 — Score Dispersion across Panel (MPAD per Case)\nRed = above median disagreement", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig7_c2_mpad.png", dpi=150)
    plt.close()
