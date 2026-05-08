# anova_test.py
# Dimension 1 - Layer 1 (VADER) - Statistical Significance Testing
# Tests whether the 5 LLMs differ significantly in mean sentiment.
# Includes: Levene's test (assumption), ANOVA (main test),
# Kruskal-Wallis (robustness), Tukey HSD (post-hoc).

import pandas as pd
import glob
import os
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =====================================================================
# STEP 1 - Load the VADER analysis file
# =====================================================================
results_folder = "results"
csv_files = glob.glob(f"{results_folder}/vader_analysis_*.csv")

if not csv_files:
    print("ERROR: No 'vader_analysis_*.csv' file found in 'results' folder.")
    exit()

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"Loading file: {latest_csv}")

df = pd.read_csv(latest_csv)
df = df.dropna(subset=["vader_compound"])
print(f"Rows analysed: {len(df)}\n")


# =====================================================================
# STEP 2 - Split scores by model into 5 groups
# =====================================================================
models = sorted(df["model_name"].unique())
groups = {m: df.loc[df["model_name"] == m, "vader_compound"].values for m in models}

print("=" * 70)
print("  GROUP SUMMARY")
print("=" * 70)
for m, scores in groups.items():
    print(f"  {m:10s}  n={len(scores):3d}   mean={scores.mean():+.4f}   std={scores.std():.4f}")


# =====================================================================
# STEP 3 - Levene's test (assumption check: equal variances?)
# =====================================================================
print("\n" + "=" * 70)
print("  ASSUMPTION CHECK - Levene's test for equal variances")
print("=" * 70)
print("  H0: All 5 groups have equal variance")
print("  If p > 0.05  --> assumption HOLDS, standard ANOVA is fine")
print("  If p < 0.05  --> variances differ, rely on Kruskal-Wallis as backup")
print()

levene_stat, levene_p = stats.levene(*groups.values())
print(f"  Levene statistic: {levene_stat:.4f}")
print(f"  p-value:          {levene_p:.6f}")

if levene_p > 0.05:
    print("  --> Variances roughly equal. Standard ANOVA is appropriate.")
else:
    print("  --> Variances differ across models. Kruskal-Wallis result is the safer headline.")


# =====================================================================
# STEP 4 - One-way ANOVA (main test)
# =====================================================================
print("\n" + "=" * 70)
print("  ONE-WAY ANOVA")
print("=" * 70)
print("  H0: All 5 LLMs produce content with the same mean sentiment")
print("  H1: At least one LLM differs")
print()

f_stat, p_val = stats.f_oneway(*groups.values())

# Effect size (eta-squared) - measures the practical size of the effect
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
# STEP 5 - Kruskal-Wallis (robustness check, non-parametric)
# =====================================================================
print("\n" + "=" * 70)
print("  ROBUSTNESS CHECK - Kruskal-Wallis test (non-parametric)")
print("=" * 70)
print("  Same logic as ANOVA, but works on ranks. Robust to non-normality.")
print()

kw_stat, kw_p = stats.kruskal(*groups.values())
print(f"  H-statistic: {kw_stat:.4f}")
print(f"  p-value:     {kw_p:.6e}")
if kw_p < 0.05:
    print("  --> Differences confirmed under non-parametric test as well.")
else:
    print("  --> Differences NOT confirmed under non-parametric test.")


# =====================================================================
# STEP 6 - Tukey HSD post-hoc (which pairs of models differ?)
# =====================================================================
print("\n" + "=" * 70)
print("  POST-HOC TEST - Tukey HSD (pairwise comparisons)")
print("  Reads: 'reject=True' means that pair of models differs significantly.")
print("=" * 70)
print()

tukey = pairwise_tukeyhsd(endog=df["vader_compound"],
                          groups=df["model_name"],
                          alpha=0.05)
print(tukey)


print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)