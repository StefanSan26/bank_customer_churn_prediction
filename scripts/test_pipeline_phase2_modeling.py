#!/usr/bin/env python3
"""Smoke test: load processed data -> train (small) -> evaluate."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

processed_path = os.path.join(ROOT, "data", "processed", "bank_churn_processed.csv")
if not os.path.exists(processed_path):
    processed_path = os.path.join(ROOT, "data", "train.csv")

from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    print("=== Phase 2: Modeling ===")
    if not os.path.exists(processed_path):
        print(f"Skip: {processed_path} not found. Run prepare_processed_data.py or run_pipeline.py first.")
        return
    df = pd.read_csv(processed_path)
    X, y = build_features(df)
    if y is None:
        print("No Exited column; skip Phase 2.")
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Small subset for quick test
    n = min(500, len(X_train))
    X_train, y_train = X_train.iloc[:n], y_train.iloc[:n]
    model = train_model(X_train, y_train, log_to_mlflow=False, verbose=0)
    metrics = evaluate_model(model, X_test, y_test, log_to_mlflow=False)
    print("Metrics:", metrics)
    print("Phase 2 OK.")


if __name__ == "__main__":
    main()
