import pandas as pd
import numpy as np
import train_open_model

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Running Diagnostic...")
df = pd.read_parquet("data/project_c_samples.parquet")
processed, cols = train_open_model.preprocess_data(df)

print("\n--- Feature Stats ---")
print(processed[cols].describe())

print("\n--- Head ---")
print(processed[cols].head())

print("\n--- Correlation with Target ---")
# Add target back for correlation
processed['open'] = df['open']
print(processed[cols + ['open']].corr()['open'].sort_values())
