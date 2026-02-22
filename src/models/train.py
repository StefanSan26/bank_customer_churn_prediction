"""Train CatBoost classifier for bank churn prediction."""
import logging
from typing import Any, Optional

import mlflow
import pandas as pd
from catboost import CatBoostClassifier

DEFAULT_PARAMS = {
    "subsample": 0.8,
    "learning_rate": 0.1,
    "l2_leaf_reg": 1,
    "depth": 4,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict[str, Any]] = None,
    log_to_mlflow: bool = True,
    verbose: int = 0,
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier and optionally log to MLflow.

    Args:
        X_train: Training features.
        y_train: Training target.
        params: CatBoost hyperparameters (defaults used if None).
        log_to_mlflow: If True, log params, metrics, and model to the active MLflow run.
        verbose: CatBoost verbosity (0 = silent).

    Returns:
        Fitted CatBoostClassifier.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = CatBoostClassifier(**p)
    model.fit(X_train, y_train, verbose=verbose)

    if log_to_mlflow:
        mlflow.log_params(p)
        mlflow.catboost.log_model(model, "model")

    return model
