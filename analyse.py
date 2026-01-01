import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT_DIR = Path("sweep_results")      # change if needed
CSV_PATTERN = "**/summary.csv"
# RUNNING_SET = "davis_novel-pair"      # <<< FILTER KEYWORD
# RUNNING_SET = "davis_novel-prot"      # <<< FILTER KEYWORD
# RUNNING_SET = "davis_novel-drug"      # <<< FILTER KEYWORD
RUNNING_SET = "_davis_warm_b256"      # <<< FILTER KEYWORD

METRICS = [
    "mse",
    "rmse",
    "ci",
    "r2",
    "pearson",
    "spearman",
    "best_epoch",
    "best_valid_mse",
    "final_entropy",
]

# =========================
# LOAD FILTERED CSVs
# =========================
dfs = []
for csv_path in ROOT_DIR.glob(CSV_PATTERN):
    if RUNNING_SET not in str(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df["source"] = str(csv_path)      # optional
    df["running_set"] = RUNNING_SET   # optional
    dfs.append(df)

if not dfs:
    raise RuntimeError(f"No summary.csv files found for running set '{RUNNING_SET}'")

data = pd.concat(dfs, ignore_index=True)

# =========================
# TYPE CAST
# =========================
for col in METRICS:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# =========================
# AGGREGATE PER MODEL
# =========================
agg = (
    data
    .groupby("config")[METRICS]
    .agg(["mean", "std"])
    .reset_index()
)

# Flatten columns
agg.columns = [
    "config" if c[0] == "config" else f"{c[0]}_{c[1]}"
    for c in agg.columns
]

# Optional sorting
agg = agg.sort_values("mse_mean")

# =========================
# OUTPUT
# =========================
print(agg)
dest_folder="analysis_results"
Path(dest_folder).mkdir(parents=True, exist_ok=True)
agg.to_csv(Path(dest_folder) / f"aggregated_results_{RUNNING_SET}.csv", index=False)
