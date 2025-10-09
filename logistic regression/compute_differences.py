import pandas as pd

# Load the datasets
teams_df = pd.read_csv("team_summary.csv")
league_df = pd.read_csv("league_averages.csv")

# Columns to exclude from difference calculation
exclude_cols = ["age", "attend_g"]
non_stat_cols = ["season", "team", "playoffs"]
stat_cols = [col for col in teams_df.columns if col not in non_stat_cols]

# Keep only performance-related stats (exclude age & attendance)
perf_stat_cols = [col for col in stat_cols if col not in exclude_cols]

# Merge team stats with league averages by season
merged = pd.merge(
    teams_df,
    league_df[["season"] + perf_stat_cols],  # only season + stat cols from league averages
    on="season",
    suffixes=("", "_league")
)

# Compute differences only for performance stats
diffs_perf = merged[["season", "team", "playoffs"]].copy()
for col in perf_stat_cols:
    diffs_perf[f"{col}_diff"] = merged[col] - merged[f"{col}_league"]

# Save to new CSV
diffs_perf.to_csv("team_differences_performance.csv", index=False)

print("Differences saved to team_differences_performance.csv")
print(diffs_perf.head())
