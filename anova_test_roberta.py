# anova_test_roberta.py
# Dimension 1 - Layer 2 (RoBERTa) - Statistical Significance Testing
# Same pipeline as anova_test.py but applied to roberta_score.

import pandas as pd
import glob
import os
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =====================================================================
# STEP 1 - Load the latest two-layer file (VADER + RoBERTa scores)
# =====================================================================
results_folder = "results"
csv_files = glob.glob(f"{results_folder}/sentiment_analysis_full_*.csv")

if not csv_files:
    print("ERROR: No 'sentiment_analysis_full_*.csv' file found.")
    exit()

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"Loading file: {latest_csv}")

df = pd.read_csv(latest_csv)
df = df.dropna(subset=["roberta_score"])
print(f"Rows analysed: {len(df)}\n")


# =====================================================================
# STEP 2 - Split scores by model into 5 groups
# =====================================================================
models = sorted(df["model_name"].unique())
groups = {m: df.loc[df["model_name"] == m, "roberta_score"].values for m in models}

print("=" * 70)
print("  GROUP SUMMARY (RoBERTa)")
print("=" * 70)
for m, scores in groups.items():
    print(f"  {m:10s}  n={len(scores):3d}   mean={scores.mean():+.4f}   std={scores.std():.4f}")


# =====================================================================
# STEP 3 - Levene's test
# =====================================================================
print("\n" + "=" * 70)
print("  ASSUMPTION CHECK - Levene's test for equal variances")
print("=" * 70)

levene_stat, levene_p = stats.levene(*groups.values())
print(f"  Levene statistic: {levene_stat:.4f}")
print(f"  p-value:          {levene_p:.6f}")

if levene_p > 0.05:
    print("  --> Variances roughly equal. Standard ANOVA appropriate.")
else:
    print("  --> Variances differ. Kruskal-Wallis will be the safer headline.")


# =====================================================================
# STEP 4 - One-way ANOVA
# =====================================================================
print("\n" + "=" * 70)
print("  ONE-WAY ANOVA  (on roberta_score)")
print("=" * 70)
print("  H0: All 5 LLMs produce the same mean RoBERTa sentiment")
print("  H1: At least one LLM differs")
print()

f_stat, p_val = stats.f_oneway(*groups.values())

all_scores = np.concatenate(list(groups.values()))
ss_total = ((all_scores - all_scores.mean()) ** 2).sum()
ss_between = sum(len(g) * (g.mean() - all_scores.mean()) ** 2 for g in groups.values())
eta_squared = ss_between / ss_total

print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value:     {p_val:.6e}")
print(f"  Eta-squared: {eta_squared:.4f}  (effect size)")
print(f"     Cohen benchmarks:  0.01 = small,  0.06 = medium,  0.14 = large")

if p_val < 0.001:
    print("  --> p < 0.001 : differences are HIGHLY significant.")
elif p_val < 0.01:
    print("  --> p < 0.01  : differences are very significant.")
elif p_val < 0.05:
    print("  --> p < 0.05  : differences are significant.")
else:
    print("  --> p > 0.05  : differences are NOT statistically significant.")


# =====================================================================
# STEP 5 - Kruskal-Wallis robustness check
# =====================================================================
print("\n" + "=" * 70)
print("  ROBUSTNESS CHECK - Kruskal-Wallis test (non-parametric)")
print("=" * 70)
kw_stat, kw_p = stats.kruskal(*groups.values())
print(f"  H-statistic: {kw_stat:.4f}")
print(f"  p-value:     {kw_p:.6e}")
if kw_p < 0.05:
    print("  --> Differences confirmed under non-parametric test as well.")
else:
    print("  --> Differences NOT confirmed under non-parametric test.")


# =====================================================================
# STEP 6 - Tukey HSD post-hoc
# =====================================================================
print("\n" + "=" * 70)
print("  POST-HOC TEST - Tukey HSD (pairwise comparisons on roberta_score)")
print("=" * 70)
tukey = pairwise_tukeyhsd(endog=df["roberta_score"],
                          groups=df["model_name"],
                          alpha=0.05)
print(tukey)

print("\n" + "=" * 70)
print("  RoBERTa STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)
