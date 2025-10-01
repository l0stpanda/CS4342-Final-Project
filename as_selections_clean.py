import pandas as pd

def clean_all_star_selections(input_path: str, output_path: str):
    # --- Load CSV ---
    df = pd.read_csv("/Users/benzeng/Desktop/Ben/machine learning project/all_star_selections.csv")
    
    # --- Filter to past 10 seasons ---
    current_year = 2025
    cutoff = 2015
    df_recent = df[df["season"] >= cutoff].copy()
    
    # --- Drop unwanted columns ---
    cols_to_drop = ["lg", "replaced"]
    df_recent = df_recent.drop(columns=[c for c in cols_to_drop if c in df_recent.columns])
    
    # --- Reset index ---
    df_recent = df_recent.reset_index(drop=True)
    
    # Save cleaned CSV
    df_recent.to_csv(output_path, index=False)
    print(f"Saved {len(df_recent)} records from seasons {cutoff}-{current_year} to {output_path}")

if __name__ == "__main__":
    input_csv = "all_star_selections.csv"
    output_csv = "all_star_selections_cleaned.csv"
    clean_all_star_selections(input_csv, output_csv)
