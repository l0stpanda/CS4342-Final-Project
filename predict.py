import argparse
import pandas as pd
import joblib
import numpy as np

def main(args):
    # Load trained pipeline (imputer + RF)
    model = joblib.load(args.model)

    # Option A: predict for a row taken from a CSV of one/many teams
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        # Try to drop non-numeric columns the model didn't train on
        # (the pipeline will select numeric columns internally, but keeping schema close helps)
        # NOTE: Do NOT include the target column.
        if args.target and args.target in df.columns:
            df = df.drop(columns=[args.target])
        # Predict
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= args.threshold).astype(int)
        out = df.copy()
        out["playoff_prob"] = probs
        out["prediction"] = preds
        print(out[["playoff_prob", "prediction"]])
        return

    # Option B: grab a single row from your training CSV by index (handy for quick tests)
    if args.train_csv and args.row_index is not None:
        train = pd.read_csv(args.train_csv)
        # Drop the target if present; the training pipeline learned with numeric-only features
        if args.target and args.target in train.columns:
            train = train.drop(columns=[args.target])
        row = train.iloc[[args.row_index]]  # keep as DataFrame
        probs = model.predict_proba(row)[:, 1]
        pred = int(probs[0] >= args.threshold)
        print(f"Probability of making playoffs: {probs[0]:.3f}")
        print("Prediction:", "MAKE (1)" if pred == 1 else "MISS (0)")
        return

    print("Provide --input-csv path OR --train-csv with --row-index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/random_forest.joblib")
    parser.add_argument("--input-csv", type=str, help="CSV with one or more teams (no target column).")
    parser.add_argument("--train-csv", type=str, help="Your original training CSV (to test a row by index).")
    parser.add_argument("--row-index", type=int, help="Row index from --train-csv to predict.")
    parser.add_argument("--target", type=str, default="playoffs", help="Name of target column (if present in CSV).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class label.")
    args = parser.parse_args()
    main(args)
