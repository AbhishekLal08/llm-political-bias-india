# compare_vader_roberta.py
# Dimension 1 - Final Comparison Chart
# Single figure with two panels showing VADER vs RoBERTa results
# side by side per model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# =====================================================================
# STEP 1 - Load the two-layer file
# =====================================================================
results_folder = "results"
figures_folder = "figures"
os.makedirs(figures_folder, exist_ok=True)

csv_files = glob.glob(f"{results_folder}/sentiment_analysis_full_*.csv")
if not csv_files:
    print("ERROR: No 'sentiment_analysis_full_*.csv' file found.")
    exit()

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"Loading file: {latest_csv}")
df = pd.read_csv(latest_csv)


# =====================================================================
# STEP 2 - Compute per-model summaries for each layer
# =====================================================================
# Order models from most positive to most negative based on VADER mean
order = (df.groupby("model_name")["vader_compound"]
           .mean().sort_values(ascending=False).index.tolist())

vader_means = df.groupby("model_name")["vader_compound"].mean().loc[order]
roberta_means = df.groupby("model_name")["roberta_score"].mean().loc[order]

# Label percentages per layer
def pct_table(label_col):
    counts = df.groupby(["model_name", label_col]).size().unstack(fill_value=0)
    for lab in ["Positive", "Neutral", "Negative"]:
        if lab not in counts.columns:
            counts[lab] = 0
    counts = counts[["Positive", "Neutral", "Negative"]].loc[order]
    return counts.div(counts.sum(axis=1), axis=0) * 100

vader_pct = pct_table("vader_label")
roberta_pct = pct_table("roberta_label")


# =====================================================================
# CHART - Two-panel comparison figure
# Panel A: Mean compound score side by side
# Panel B: Stacked label percentages, side by side
# =====================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7))
plt.rcParams.update({"font.size": 11})

# ---- PANEL A: Mean scores -----------------------------------------
x = np.arange(len(order))
width = 0.38
axA.bar(x - width/2, vader_means.values, width,
        color="#3a86ff", edgecolor="black", linewidth=0.8, label="VADER (Layer 1)")
axA.bar(x + width/2, roberta_means.values, width,
        color="#ff6b6b", edgecolor="black", linewidth=0.8, label="RoBERTa (Layer 2)")

# Annotate bars with values
for i, v in enumerate(vader_means.values):
    offset = 0.02 if v >= 0 else -0.04
    axA.text(i - width/2, v + offset, f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")
for i, v in enumerate(roberta_means.values):
    offset = 0.02 if v >= 0 else -0.04
    axA.text(i + width/2, v + offset, f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")

axA.axhline(0, color="black", linewidth=0.8)
axA.set_xticks(x)
axA.set_xticklabels(order)
axA.set_ylabel("Mean Sentiment Score  (-1 to +1)")
axA.set_title("A.  Mean Sentiment Score: VADER vs RoBERTa", fontweight="bold", pad=12)
axA.set_ylim(-0.35, 0.45)
axA.legend(loc="upper right", framealpha=0.95)
axA.grid(axis="y", alpha=0.3)

# ---- PANEL B: Stacked label percentages -----------------------------
label_colours = {"Positive": "#2ca02c", "Neutral": "#bbbbbb", "Negative": "#d62728"}

# We'll have 2 stacked bars per model (VADER and RoBERTa)
positions_v = x - width/2
positions_r = x + width/2

# VADER stacks
bottom = np.zeros(len(order))
for label in ["Positive", "Neutral", "Negative"]:
    vals = vader_pct[label].values
    axB.bar(positions_v, vals, width, bottom=bottom,
            color=label_colours[label], edgecolor="white", linewidth=1)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 6:
            axB.text(positions_v[i], b + v/2, f"{v:.0f}%",
                     ha="center", va="center", color="white",
                     fontweight="bold", fontsize=9)
    bottom += vals

# RoBERTa stacks (with legend handles created here)
bottom = np.zeros(len(order))
legend_handles = {}
for label in ["Positive", "Neutral", "Negative"]:
    vals = roberta_pct[label].values
    bars = axB.bar(positions_r, vals, width, bottom=bottom,
                   color=label_colours[label], edgecolor="white", linewidth=1,
                   label=label)
    legend_handles[label] = bars
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 6:
            axB.text(positions_r[i], b + v/2, f"{v:.0f}%",
                     ha="center", va="center", color="white",
                     fontweight="bold", fontsize=9)
    bottom += vals

# Add 'V' and 'R' labels under each pair of bars
for i in range(len(order)):
    axB.text(positions_v[i], -3, "V", ha="center", fontsize=9, fontweight="bold", color="#3a86ff")
    axB.text(positions_r[i], -3, "R", ha="center", fontsize=9, fontweight="bold", color="#ff6b6b")

axB.set_xticks(x)
axB.set_xticklabels(order)
axB.set_ylabel("Percentage of Responses")
axB.set_title("B.  Label Distribution: VADER (V) vs RoBERTa (R)", fontweight="bold", pad=12)
axB.set_ylim(-7, 105)
axB.legend(loc="upper right", framealpha=0.95)
axB.grid(axis="y", alpha=0.3)

fig.suptitle("Layer 1 (VADER) vs Layer 2 (RoBERTa) Sentiment Comparison",
             fontweight="bold", fontsize=14, y=1.00)
plt.tight_layout()

out = f"{figures_folder}/comparison_vader_vs_roberta.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# =====================================================================
# Print the comparison table to terminal as well
# =====================================================================
print("\n" + "=" * 80)
print("  SIDE-BY-SIDE COMPARISON TABLE")
print("=" * 80)
print(f"{'Model':10s} | {'VADER mean':>11s} | {'RoBERTa mean':>13s} | "
      f"{'V Pos%':>7s} | {'V Neu%':>7s} | {'V Neg%':>7s} | "
      f"{'R Pos%':>7s} | {'R Neu%':>7s} | {'R Neg%':>7s}")
print("-" * 80)
for m in order:
    print(f"{m:10s} | {vader_means[m]:+11.3f} | {roberta_means[m]:+13.3f} | "
          f"{vader_pct.loc[m, 'Positive']:7.1f} | {vader_pct.loc[m, 'Neutral']:7.1f} | {vader_pct.loc[m, 'Negative']:7.1f} | "
          f"{roberta_pct.loc[m, 'Positive']:7.1f} | {roberta_pct.loc[m, 'Neutral']:7.1f} | {roberta_pct.loc[m, 'Negative']:7.1f}")

print("\n" + "=" * 80)
print("  COMPARISON COMPLETE")
print("=" * 80)
