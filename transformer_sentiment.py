# transformer_sentiment.py
# Dimension 1 - Sentiment Analysis - Layer 2 (Transformer-based)
# Uses cardiffnlp/twitter-roberta-base-sentiment-latest
# Chunks long responses (>510 tokens) into pieces, scores each chunk,
# and averages the probabilities to produce one score per response.

import pandas as pd
import numpy as np
import glob
import os
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


# =====================================================================
# STEP 1 - Load the latest VADER analysis file
# =====================================================================
results_folder = "results"
csv_files = glob.glob(f"{results_folder}/vader_analysis_*.csv")

if not csv_files:
    print("ERROR: No VADER analysis CSV found. Run vader_analysis.py first.")
    exit()

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"Loading file: {latest_csv}")

df = pd.read_csv(latest_csv)
print(f"Loaded {len(df)} rows.\n")


# =====================================================================
# STEP 2 - Load the transformer model and tokenizer
# (Reuses the cached download from the first attempt - no re-download)
# =====================================================================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Special token IDs - we'll add these manually around each chunk
CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id

# Label mapping for this specific model
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

print("Model loaded successfully.\n")


# =====================================================================
# STEP 3 - Define the scoring function with chunking for long text
# (FIXED: manually wraps chunks with [CLS] and [SEP] tokens; universal
#  across all transformers/tokenizer versions)
# =====================================================================
CHUNK_SIZE = 510   # leave room for CLS and SEP

def score_response(text):
    """Tokenize text, split into chunks if long, score each chunk,
       average probabilities to produce a single set of probs."""
    if pd.isna(text) or str(text).strip() == "" or str(text).startswith("ERROR"):
        return None

    text = str(text)

    # Tokenize the entire text without special tokens or truncation
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    if len(tokens) == 0:
        return None

    # Split tokens into chunks of CHUNK_SIZE
    chunks = [tokens[i:i + CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]

    # Score each chunk
    chunk_probs = []
    for chunk in chunks:
        # Manually wrap with [CLS] and [SEP] - works on every tokenizer
        input_ids = [CLS_ID] + chunk + [SEP_ID]
        input_tensor = torch.tensor([input_ids])
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            output = model(input_ids=input_tensor, attention_mask=attention_mask)

        probs = softmax(output.logits.numpy()[0])
        chunk_probs.append(probs)

    # Average probabilities across all chunks
    avg_probs = np.mean(chunk_probs, axis=0)
    return {
        "neg": float(avg_probs[0]),
        "neu": float(avg_probs[1]),
        "pos": float(avg_probs[2]),
        "n_chunks": len(chunks),
    }


# =====================================================================
# STEP 4 - Run the model on every response with progress updates
# =====================================================================
print("=" * 70)
print("  Scoring 450 responses with RoBERTa")
print("  (this typically takes 10-25 minutes on CPU)")
print("=" * 70)

results = []
start_time = time.time()
report_every = 25

for i, row in df.iterrows():
    res = score_response(row["response_text"])
    results.append(res)

    if (i + 1) % report_every == 0 or (i + 1) == len(df):
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(df) - (i + 1)) / rate if rate > 0 else 0
        print(f"  Processed {i + 1:3d}/{len(df)}   "
              f"elapsed: {elapsed/60:.1f} min   "
              f"ETA: {eta/60:.1f} min")

print(f"\nScoring complete in {(time.time() - start_time)/60:.1f} minutes.\n")


# =====================================================================
# STEP 5 - Add the new columns to the dataframe
# =====================================================================
df["roberta_neg"]      = [r["neg"] if r else None for r in results]
df["roberta_neu"]      = [r["neu"] if r else None for r in results]
df["roberta_pos"]      = [r["pos"] if r else None for r in results]
df["roberta_n_chunks"] = [r["n_chunks"] if r else None for r in results]

df["roberta_score"] = df["roberta_pos"] - df["roberta_neg"]

def best_label(row):
    if pd.isna(row["roberta_pos"]):
        return "ERROR"
    probs = [row["roberta_neg"], row["roberta_neu"], row["roberta_pos"]]
    return LABEL_MAP[int(np.argmax(probs))]

df["roberta_label"] = df.apply(best_label, axis=1)


# =====================================================================
# STEP 6 - Save the final two-layer file
# =====================================================================
date_string = datetime.now().strftime("%Y-%m-%d_%H%M")
output_csv  = f"{results_folder}/sentiment_analysis_full_{date_string}.csv"
output_xlsx = f"{results_folder}/sentiment_analysis_full_{date_string}.xlsx"

df.to_csv(output_csv, index=False, encoding="utf-8")
df.to_excel(output_xlsx, index=False, engine="openpyxl")

print(f"Results saved to:")
print(f"  CSV:   {output_csv}")
print(f"  Excel: {output_xlsx}\n")


# =====================================================================
# STEP 7 - Quick on-screen summary
# =====================================================================
print("=" * 70)
print("  AVERAGE RoBERTa SCORE PER MODEL")
print("  (range: -1 = negative, 0 = neutral, +1 = positive)")
print("=" * 70)
summary = (df.groupby("model_name")["roberta_score"]
             .agg(["mean", "median", "std"]).round(4)
             .sort_values("mean", ascending=False))
print(summary)

print("\n" + "=" * 70)
print("  RoBERTa LABEL DISTRIBUTION PER MODEL")
print("=" * 70)
label_dist = df.groupby(["model_name", "roberta_label"]).size().unstack(fill_value=0)
order = [c for c in ["Positive", "Neutral", "Negative"] if c in label_dist.columns]
print(label_dist[order])

print("\n" + "=" * 70)
print("  PERCENTAGE BREAKDOWN PER MODEL")
print("=" * 70)
pct = label_dist[order].div(label_dist[order].sum(axis=1), axis=0).mul(100).round(1)
print(pct)

print("\n" + "=" * 70)
print("  LAYER 2 (TRANSFORMER) ANALYSIS COMPLETE")
print("=" * 70)
