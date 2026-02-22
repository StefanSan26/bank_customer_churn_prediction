"""Evaluate bank churn model: metrics and optional MLflow logging."""
from typing import Optional

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_to_mlflow: bool = True,
) -> dict:
    """
    Compute accuracy, precision, recall, and optionally ROC AUC; optionally log to MLflow.

    Args:
        model: Fitted classifier with .predict and .predict_proba.
        X_test: Test features.
        y_test: True labels.
        log_to_mlflow: If True, log metrics to the active MLflow run.

    Returns:
        Dict with keys: accuracy, precision, recall, roc_auc (if predict_proba available).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    try:
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
    except Exception:
        pass
    if log_to_mlflow:
        mlflow.log_metrics(metrics)
    return metrics
