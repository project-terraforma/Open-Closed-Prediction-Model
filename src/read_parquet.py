import pandas as pd
import json

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

print("Loading parquet...")
df = pd.read_parquet("data/project_c_samples.parquet")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# -------------------------
# SAFE JSON EXPORT
# -------------------------

print("\nExporting JSON (safe mode)...")

records = df.to_dict(orient="records")

with open("data/project_c_raw.json", "w", encoding="utf-8", errors="replace") as f:
    json.dump(
        records,
        f,
        indent=2,
        ensure_ascii=False,
        default=str  # <- THIS IS THE KEY
    )

print("JSON export complete: project_c_raw.json")

# -------------------------
# SAFE CSV EXPORT
# -------------------------

print("Exporting CSV...")
df.to_csv(
    "data/project_c_raw.csv",
    index=False,
    encoding="utf-8",
    errors="replace"
)

print("CSV export complete: project_c_raw.csv")
