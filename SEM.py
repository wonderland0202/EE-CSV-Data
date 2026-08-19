import pandas as pd
import numpy as np
from scipy import stats

# Load raw trial CSV
df = pd.read_csv("ee_individual_trials_raw.csv")

# Filter for Paris
paris_df = df[df["tier_name"].str.contains("Paris")]
N = len(paris_df)  # 500

for L in [4, 8, 16]:
    times = paris_df[f"alt_L{L}_time_ms"]
    
    mean = np.mean(times)
    sem = stats.sem(times) # s / sqrt(N)
    
    # 95% Confidence Interval margin
    margin = sem * stats.t.ppf((1 + 0.95) / 2., N - 1)
    
    print(f"ALT L={L:2d}: Mean = {mean:.2f} ms ± {margin:.2f} ms | 95% CI: [{mean - margin:.2f}, {mean + margin:.2f}]")
