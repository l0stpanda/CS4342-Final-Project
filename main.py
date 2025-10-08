import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

def plot_confusion_matrix(cm, title, outpath):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks([0,1],[0,1]); plt.yticks([0,1],[0,1])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_feature_importances(importances, feature_names, outpath, top_k=20):
    order = np.argsort(importances)[::-1][:top_k]
    top_feats = np.array(feature_names)[order]
    top_vals = importances[order]

    plt.figure(figsize=(8,6))
    plt.barh(top_feats[::-1], top_vals[::-1])
    plt.title(f"Random Forest Feature Importances (Top {len(top_feats)})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main(args):
    # I/O setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # load data
    df = pd.read_csv(args.csv)

    # Check target
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")
        return

    # Target: convert boolean to 0/1 if needed
    y_raw = df[args.target]
    if y_raw.dtype == bool or set(pd.unique(y_raw)).issubset({True, False}):
        y = y_raw.astype(int)
    else:
        y = y_raw

    # Features: keep numeric only; drop clearly non-predictive identifiers if present
    X = df.drop(columns=[args.target, "team"], errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # pipeline
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), numeric_cols)],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced"  # helps if classes are imbalanced
    )

    pipe = Pipeline([("pre", pre), ("rf", rf)])

    # Light hyperparameter search to get a solid RF 
    # Keeps runtime reasonable but improves performance vs defaults
    param_dist = {
        "rf__n_estimators": randint(250, 600),
        "rf__max_depth": randint(3, 20),
        "rf__min_samples_split": randint(2, 20),
        "rf__min_samples_leaf": randint(1, 10),
        "rf__max_features": uniform(0.3, 0.7),  # fraction of features
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.search_iters,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        random_state=args.random_state,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate 
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    try:
        proba = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
    except Exception:
        roc_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=[0,1])
    plot_confusion_matrix(cm, "Confusion Matrix - RandomForest", "artifacts/confusion_matrix_rf.png")

    # Classification report (full)
    report_text = classification_report(y_test, preds, digits=3)
    with open("artifacts/classification_report_rf.txt", "w") as f:
        f.write(report_text)

    # Save metrics & best params
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "best_params": search.best_params_
    }
    with open("artifacts/metrics_rf.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Feature importances
    # Get the RF inside the pipeline and save importances + plot
    rf_fitted = best_model.named_steps["rf"]
    imputer = best_model.named_steps["pre"].named_transformers_["num"]
    feature_names_after = numeric_cols  # imputer keeps column order

    importances = rf_fitted.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names_after, "importance": importances}) \
            .sort_values("importance", ascending=False)
    fi_df.to_csv("artifacts/rf_feature_importances.csv", index=False)
    plot_feature_importances(importances, feature_names_after, "artifacts/rf_feature_importances.png")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "random_forest.joblib")
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV")
    parser.add_argument("--target", type=str, default="playoffs", help="Target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--search-iters", type=int, default=30, help="RandomizedSearchCV iterations")
    args = parser.parse_args()
    main(args)
