# vader_visualisations.py
# Dimension 1 - Layer 1 (VADER) - Visualisations
# Generates 4 thesis-quality charts from the VADER analysis CSV:
#   1. Box plot of compound scores per model
#   2. Bar chart of mean compound score per model (with 95% CI error bars)
#   3. Stacked percentage bar of sentiment labels per model
#   4. Histograms of compound scores per model (shows the bimodal shape)
# All figures saved to a 'figures/' folder at 300 DPI.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


# =====================================================================
# STEP 1 - Load the VADER analysis file
# =====================================================================
results_folder = "results"
figures_folder = "figures"
os.makedirs(figures_folder, exist_ok=True)

csv_files = glob.glob(f"{results_folder}/vader_analysis_*.csv")
if not csv_files:
    print("ERROR: No 'vader_analysis_*.csv' file found.")
    exit()

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"Loading file: {latest_csv}")
df = pd.read_csv(latest_csv)
df = df.dropna(subset=["vader_compound"])


# =====================================================================
# STEP 2 - Order the models from most positive to most negative
#          (so all charts are consistent and easy to read)
# =====================================================================
order = (df.groupby("model_name")["vader_compound"].mean()
           .sort_values(ascending=False).index.tolist())

# A consistent colour palette (one colour per model)
palette = {
    "ChatGPT":  "#10a37f",   # OpenAI green
    "Claude":   "#cc785c",   # Anthropic orange
    "Gemini":   "#4285f4",   # Google blue
    "DeepSeek": "#5e60ce",   # purple
    "Grok":     "#1d1d1d",   # near-black
}

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})


# =====================================================================
# CHART 1 - Box plot
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=df, x="model_name", y="vader_compound", order=order,
            palette=palette, ax=ax, width=0.55, fliersize=3)
ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax.set_title("VADER Compound Sentiment Score Distribution by LLM",
             fontweight="bold", pad=15)
ax.set_xlabel("Large Language Model")
ax.set_ylabel("VADER Compound Score  (-1 = very negative,  +1 = very positive)")
ax.set_ylim(-1.05, 1.05)

# Add median values on top of each box
medians = df.groupby("model_name")["vader_compound"].median()
for i, m in enumerate(order):
    ax.text(i, 1.01, f"med={medians[m]:+.2f}", ha="center", fontsize=9, color="grey")

plt.tight_layout()
out1 = f"{figures_folder}/vader_01_boxplot.png"
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()


# =====================================================================
# CHART 2 - Bar chart of means with 95% confidence intervals
# =====================================================================
summary = df.groupby("model_name")["vader_compound"].agg(["mean", "std", "count"])
summary["se"] = summary["std"] / np.sqrt(summary["count"])
summary["ci95"] = 1.96 * summary["se"]
summary = summary.loc[order]  # keep consistent order

fig, ax = plt.subplots(figsize=(9, 6))
colours = [palette[m] for m in order]
bars = ax.bar(summary.index, summary["mean"], yerr=summary["ci95"],
              color=colours, capsize=6, edgecolor="black", linewidth=0.8)

# Annotate each bar with its mean
for bar, mean in zip(bars, summary["mean"]):
    y = bar.get_height()
    offset = 0.03 if y >= 0 else -0.05
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f"{mean:+.3f}", ha="center", fontsize=10, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean VADER Compound Score by LLM  (with 95% confidence intervals)",
             fontweight="bold", pad=15)
ax.set_xlabel("Large Language Model")
ax.set_ylabel("Mean VADER Compound Score")
ax.set_ylim(-0.4, 0.5)
plt.tight_layout()
out2 = f"{figures_folder}/vader_02_means_with_ci.png"
plt.savefig(out2, dpi=300, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()


# =====================================================================
# CHART 3 - Stacked percentage bar of sentiment labels per model
# =====================================================================
counts = df.groupby(["model_name", "vader_label"]).size().unstack(fill_value=0)
# Make sure all three labels exist as columns
for lab in ["Positive", "Neutral", "Negative"]:
    if lab not in counts.columns:
        counts[lab] = 0
counts = counts[["Positive", "Neutral", "Negative"]].loc[order]
pct = counts.div(counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 6))
label_colours = {"Positive": "#2ca02c", "Neutral": "#bbbbbb", "Negative": "#d62728"}
bottom = np.zeros(len(pct))
for label in ["Positive", "Neutral", "Negative"]:
    vals = pct[label].values
    ax.bar(pct.index, vals, bottom=bottom, color=label_colours[label],
           label=label, edgecolor="white", linewidth=1)
    # Add labels inside each segment
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:  # only label if segment is big enough
            ax.text(i, b + v/2, f"{v:.1f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
    bottom += vals

ax.set_title("Sentiment Label Distribution by LLM  (% of responses)",
             fontweight="bold", pad=15)
ax.set_xlabel("Large Language Model")
ax.set_ylabel("Percentage of Responses")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", framealpha=0.95)
plt.tight_layout()
out3 = f"{figures_folder}/vader_03_label_distribution.png"
plt.savefig(out3, dpi=300, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()


# =====================================================================
# CHART 4 - Histograms per model (reveals the bimodal U-shape)
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, m in enumerate(order):
    ax = axes[i]
    scores = df.loc[df["model_name"] == m, "vader_compound"].values
    ax.hist(scores, bins=20, color=palette[m], edgecolor="white", alpha=0.85)
    ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"mean = {scores.mean():+.2f}")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title(m, fontweight="bold")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Number of Responses")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_xlim(-1.05, 1.05)

# Hide the empty 6th subplot
axes[5].axis("off")

fig.suptitle("Distribution of VADER Compound Scores per LLM  (n = 90 each)",
             fontweight="bold", fontsize=14, y=1.00)
plt.tight_layout()
out4 = f"{figures_folder}/vader_04_histograms.png"
plt.savefig(out4, dpi=300, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close()


print()
print("=" * 70)
print(f"  ALL 4 FIGURES SAVED TO: {figures_folder}/")
print("=" * 70)