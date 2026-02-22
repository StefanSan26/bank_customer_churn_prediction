#!/usr/bin/env python3
"""
Run bank churn pipeline: load -> validate -> preprocess -> build_features -> train -> evaluate.
Logs to MLflow. Run from project root: python scripts/run_pipeline.py [--input data/train.csv]
"""
import argparse
import os
import sys

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root and src on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.validate_data import validate_bank_data


def main(args):
    mlflow_uri = args.mlflow_uri or f"file://{ROOT}/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "catboost")
        mlflow.log_param("test_size", args.test_size)

        # Load
        print("Loading data...")
        df = load_data(args.input)
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Validate
        print("Validating data...")
        is_valid, failed = validate_bank_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            raise ValueError(f"Data validation failed: {failed}")
        print("Validation passed.")

        # Preprocess
        print("Preprocessing...")
        df = preprocess_data(df)

        processed_path = os.path.join(ROOT, "data", "processed", "bank_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Saved processed data to {processed_path}")

        # Features
        print("Building features...")
        X, y = build_features(df)
        if y is None:
            raise ValueError("Target column Exited not found")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

        # Train
        print("Training model...")
        model = train_model(X_train, y_train, log_to_mlflow=True, verbose=0)

        # Evaluate
        print("Evaluating...")
        metrics = evaluate_model(model, X_test, y_test, log_to_mlflow=True)
        print("Metrics:", metrics)
        print("Pipeline finished successfully.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bank churn pipeline (load -> validate -> preprocess -> train -> evaluate)")
    p.add_argument("--input", type=str, default="data/train.csv", help="Path to training CSV")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="bank_churn_prediction")
    p.add_argument("--mlflow_uri", type=str, default=None)
    args = p.parse_args()
    main(args)
