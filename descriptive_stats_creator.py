import pandas as pd

# --- Load dataset ---
df = pd.read_csv("Team Summaries_updated1.csv")

# --- Selecting relevant columns ---
cols = ["d_rtg", "Strength of Schedule", "attend_g"]

# --- Compute descriptive statistics ---
desc_stats = df[cols].describe().T  # transpose for readability

print("Descriptive Statistics for Selected Columns:\n")
print(desc_stats)
