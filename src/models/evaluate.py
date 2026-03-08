"""Evaluate bank churn model: metrics and optional MLflow logging."""
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_to_mlflow: bool = True,
) -> dict:
    """
    Compute accuracy, precision, recall, F1, and optionally ROC AUC; optionally log to MLflow.

    When log_to_mlflow is True, also logs confusion matrix and ROC/PR curve plots as artifacts.

    Args:
        model: Fitted classifier with .predict and .predict_proba.
        X_test: Test features.
        y_test: True labels.
        log_to_mlflow: If True, log metrics and artifact plots to the active MLflow run.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc (if predict_proba available).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
    except Exception:
        pass

    if log_to_mlflow:
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            cm_path = Path(tmpdir) / "confusion_matrix.png"
            fig.savefig(cm_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # ROC and PR curves (when predict_proba available)
            if y_proba is not None and y_proba.shape[1] >= 2:
                proba_positive = y_proba[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba_positive)
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba_positive)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.plot(fpr, tpr)
                ax1.plot([0, 1], [0, 1], "k--")
                ax1.set_xlabel("False Positive Rate")
                ax1.set_ylabel("True Positive Rate")
                ax1.set_title("ROC Curve")
                ax2.plot(recall_curve, precision_curve)
                ax2.set_xlabel("Recall")
                ax2.set_ylabel("Precision")
                ax2.set_title("Precision-Recall Curve")
                fig.tight_layout()
                curves_path = Path(tmpdir) / "roc_pr_curves.png"
                fig.savefig(curves_path, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(curves_path), artifact_path="plots")

    return metrics
