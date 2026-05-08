# vader_analysis.py
# Dimension 1 - Sentiment Analysis - Layer 1 (VADER)
# Runs VADER sentiment scoring on all 450 LLM responses,
# adds new columns for the scores, and saves the result to a new file.
# The original CSV is NOT modified.

import pandas as pd
import glob
import os
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =====================================================================
# STEP 1 - Find the input CSV file in the results folder
# =====================================================================
results_folder = "results"
csv_files = glob.glob(f"{results_folder}/llm_responses_*.csv")

if not csv_files:
    print("ERROR: No 'llm_responses_*.csv' file found in the 'results' folder.")
    exit()

# If there are multiple, pick the most recently modified one
latest_csv = max(csv_files, key=os.path.getmtime)
print(f"Loading file: {latest_csv}")


# =====================================================================
# STEP 2 - Load the data
# =====================================================================
df = pd.read_csv(latest_csv)
print(f"Loaded {len(df)} rows.\n")


# =====================================================================
# STEP 3 - Initialize VADER and define a helper function
# =====================================================================
analyzer = SentimentIntensityAnalyzer()

def score_text(text):
    """Return VADER scores (pos, neg, neu, compound) for a piece of text.
       Returns (None, None, None, None) if the text is empty or an error string."""
    if pd.isna(text) or str(text).strip() == "":
        return (None, None, None, None)
    if str(text).startswith("ERROR"):
        return (None, None, None, None)
    scores = analyzer.polarity_scores(str(text))
    return (scores["pos"], scores["neg"], scores["neu"], scores["compound"])


# =====================================================================
# STEP 4 - Run VADER on every response (all 450 rows)
# =====================================================================
print("Running VADER on all responses...")

results = df["response_text"].apply(score_text)

df["vader_pos"]      = [r[0] for r in results]
df["vader_neg"]      = [r[1] for r in results]
df["vader_neu"]      = [r[2] for r in results]
df["vader_compound"] = [r[3] for r in results]


# =====================================================================
# STEP 5 - Classify each response into Positive / Negative / Neutral
# Using VADER's standard thresholds.
# =====================================================================
def classify(compound):
    if compound is None:
        return "ERROR"
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["vader_label"] = df["vader_compound"].apply(classify)


# =====================================================================
# STEP 6 - Save the enriched table to a new CSV and Excel file
# =====================================================================
date_string = datetime.now().strftime("%Y-%m-%d_%H%M")
output_csv  = f"{results_folder}/vader_analysis_{date_string}.csv"
output_xlsx = f"{results_folder}/vader_analysis_{date_string}.xlsx"

df.to_csv(output_csv, index=False, encoding="utf-8")
df.to_excel(output_xlsx, index=False, engine="openpyxl")

print(f"\nResults saved to:")
print(f"  CSV:   {output_csv}")
print(f"  Excel: {output_xlsx}")


# =====================================================================
# STEP 7 - Print a quick on-screen summary
# =====================================================================
print("\n" + "=" * 70)
print("  AVERAGE COMPOUND SCORE PER MODEL")
print("  (range: -1 = very negative, 0 = neutral, +1 = very positive)")
print("=" * 70)

summary = df.groupby("model_name")["vader_compound"].agg(["mean", "median", "std"]).round(4)
print(summary)

print("\n" + "=" * 70)
print("  SENTIMENT LABEL DISTRIBUTION PER MODEL")
print("  (how many responses each model produced as Pos / Neu / Neg)")
print("=" * 70)
label_dist = df.groupby(["model_name", "vader_label"]).size().unstack(fill_value=0)
print(label_dist)

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)