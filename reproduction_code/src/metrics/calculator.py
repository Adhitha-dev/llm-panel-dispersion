import pandas as pd
import numpy as np
from pathlib import Path
import itertools

def compute_c1_metrics(df_c1: pd.DataFrame) -> pd.DataFrame:
    """Computes C1 (within-model) metrics per case and per model."""
    results = []
    
    for (case_id, model), group in df_c1.groupby(["case_id", "model"]):
        R = len(group)
        if R == 0: continue
            
        # Outcome Level
        verdict_counts = group["parsed_verdict"].value_counts()
        max_c = verdict_counts.max()
        flip_rate = 1 - (max_c / R)
        scr = 1 - flip_rate
        
        scores = group["parsed_score"].values
        score_var = np.var(scores, ddof=1) if R > 1 else 0
        score_sd = np.sqrt(score_var)
        
        # Rubric Level Distance D(A, B)
        # 3 dimensions: market_potential, technical_feasibility, business_viability
        replicates = group.to_dict('records')
        distances = []
        
        for a, b in itertools.combinations(replicates, 2):
            d_market = abs(a["parsed_rubric_market_potential"] - b["parsed_rubric_market_potential"])
            d_tech = abs(a["parsed_rubric_technical_feasibility"] - b["parsed_rubric_technical_feasibility"])
            d_biz = abs(a["parsed_rubric_business_viability"] - b["parsed_rubric_business_viability"])
            
            d_total = (d_market + d_tech + d_biz) / (3 * 5) # normalized to [0, 1]
            distances.append(d_total)
            
        mean_rubric_dist = np.mean(distances) if distances else 0.0
        
        results.append({
            "case_id": case_id,
            "model": model,
            "condition": "C1",
            "flip_rate": flip_rate,
            "self_consistency_rate": scr,
            "score_variance": score_var,
            "score_sd": score_sd,
            "rubric_profile_distance": mean_rubric_dist,
            "R_valid": R
        })
        
    return pd.DataFrame(results)

def compute_c2_metrics(df_c2: pd.DataFrame) -> dict:
    """Computes C2 (cross-model panel) metrics per case and global."""
    results = []
    
    # 1. Per-case metrics
    for case_id, group in df_c2.groupby("case_id"):
        k = len(group)
        if k < 2: continue
        
        # Verdict Entropy
        verdict_counts = group["parsed_verdict"].value_counts(normalize=True)
        entropy = -np.sum(verdict_counts * np.log2(verdict_counts))
        
        # Score MPAD and Range
        scores = group["parsed_score"].values
        
        # MPAD
        abs_diffs = []
        for a, b in itertools.combinations(scores, 2):
            abs_diffs.append(abs(a - b))
        mpad = np.mean(abs_diffs) if abs_diffs else 0
        
        score_range = np.max(scores) - np.min(scores)
        
        results.append({
            "case_id": case_id,
            "condition": "C2",
            "verdict_entropy": entropy,
            "score_mpad": mpad,
            "score_range": score_range,
            "k_valid": k
        })
        
    case_metrics_df = pd.DataFrame(results)
    
    # 2. Global Panel Metrics (Pairwise correlation & Effective Viewpoints)
    pivot_scores = df_c2.pivot(index="case_id", columns="model", values="parsed_score")
    correlation_matrix = pivot_scores.corr(method="pearson")
    
    # Extract upper triangle of correlation matrix
    corrs = []
    models = correlation_matrix.columns
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            val = correlation_matrix.iloc[i, j]
            if not np.isnan(val):
                corrs.append(val)
                
    mean_corr = np.mean(corrs) if corrs else 0
    k_nominal = len(models)
    effective_viewpoints = k_nominal / (1 + (k_nominal - 1) * mean_corr) if k_nominal > 0 else 0
    
    # 3. Pairwise model agreement matrix (verdict matches)
    pivot_verdicts = df_c2.pivot(index="case_id", columns="model", values="parsed_verdict")
    
    # Create empty dataframe for agreement matrix
    agreement_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                agreement_matrix.loc[m1, m2] = 1.0
            else:
                # Count matching non-null verdicts
                valid_cases = pivot_verdicts[[m1, m2]].dropna()
                if len(valid_cases) > 0:
                    matches = (valid_cases[m1] == valid_cases[m2]).sum()
                    agreement_matrix.loc[m1, m2] = matches / len(valid_cases)
                else:
                    agreement_matrix.loc[m1, m2] = np.nan
    
    return {
        "per_case": case_metrics_df,
        "correlation_matrix": correlation_matrix,
        "agreement_matrix": agreement_matrix,
        "effective_viewpoints": effective_viewpoints,
        "mean_pairwise_correlation": mean_corr
    }

def generate_four_pattern_grid(df_c1_metrics: pd.DataFrame, df_c2_metrics: pd.DataFrame, 
                              c1_threshold: float = None, c2_threshold: float = None) -> pd.DataFrame:
    """Classifies cases into the 4-pattern grid."""
    
    # Merge C1 and C2 metrics on case_id
    # If C1 has multiple models, we might take the mean C1 metrics across models for the case, or pick a primary model.
    # We'll take the mean for simplicity
    c1_agg = df_c1_metrics.groupby("case_id")[["score_sd"]].mean().reset_index()
    c2_agg = df_c2_metrics[["case_id", "score_mpad"]]
    
    merged = pd.merge(c1_agg, c2_agg, on="case_id", how="inner")
    
    if c1_threshold is None:
        c1_threshold = merged["score_sd"].median()
    if c2_threshold is None:
        c2_threshold = merged["score_mpad"].median()
        
    def classify(row):
        c1_low = row["score_sd"] <= c1_threshold
        c2_low = row["score_mpad"] <= c2_threshold
        if c1_low and c2_low: return "Stable task; judges converge"
        if c1_low and not c2_low: return "Distinct, stable model dispositions dominate"
        if not c1_low and c2_low: return "Individually unstable, but average behavior converges"
        return "Evaluation setting itself is highly uncertain"
        
    merged["pattern"] = merged.apply(classify, axis=1)
    return merged
