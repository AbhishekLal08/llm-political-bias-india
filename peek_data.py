# peek_data.py
# A quick look at the response data to confirm it loads correctly
# and to see exactly what columns are inside.

import pandas as pd
import glob
import os

# Folder where your results are stored
results_folder = "results"

# Find all CSV files in that folder that look like LLM response files
csv_files = glob.glob(f"{results_folder}/llm_responses_*.csv")

if not csv_files:
    print("ERROR: No CSV file found in the 'results' folder.")
    print("Make sure your data file is named like 'llm_responses_YYYY-MM-DD_HHMM.csv'")
    exit()

# Pick the most recently modified file (in case there are multiple)
latest_csv = max(csv_files, key=os.path.getmtime)

print("=" * 70)
print(f"  FILE FOUND: {latest_csv}")
print("=" * 70)

# Load the data into pandas (a tool that handles tables in Python)
df = pd.read_csv(latest_csv)

print(f"  Total rows:    {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print()
print("  Column names:")
for col in df.columns:
    print(f"    - {col}")
print()

# Show how many responses we have per model (if model_name column exists)
if "model_name" in df.columns:
    print("  Responses per model:")
    counts = df["model_name"].value_counts()
    for model, count in counts.items():
        print(f"    {model:12s}: {count}")
    print()

print("=" * 70)
print("  PREVIEW OF THE FIRST ROW")
print("=" * 70)

for col in df.columns:
    value = str(df.iloc[0][col])
    # If the value is very long (like a full LLM response), truncate it
    if len(value) > 200:
        value = value[:200] + "  ...[truncated]"
    print(f"  {col}:")
    print(f"    {value}")
    print()

print("=" * 70)
print("  PEEK COMPLETE")
print("=" * 70)
