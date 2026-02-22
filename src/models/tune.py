"""Hyperparameter tuning for bank churn model (stub / simple grid)."""
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from catboost import CatBoostClassifier

from src.models.train import train_model

# Simple grid for optional tuning
TUNE_GRID: List[Dict[str, Any]] = [
    {"depth": 4, "learning_rate": 0.1},
    {"depth": 6, "learning_rate": 0.05},
    {"depth": 4, "learning_rate": 0.05},
]


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    grid: Optional[List[Dict[str, Any]]] = None,
) -> CatBoostClassifier:
    """
    Run a simple grid over params and return the best model (by accuracy on validation or train).

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Optional validation features.
        y_val: Optional validation target.
        grid: List of param dicts (default TUNE_GRID).

    Returns:
        Best CatBoostClassifier from the grid.
    """
    from sklearn.metrics import accuracy_score

    grid = grid or TUNE_GRID
    best_model = None
    best_score = -1.0
    for params in grid:
        model = train_model(X_train, y_train, params=params, log_to_mlflow=False, verbose=0)
        if X_val is not None and y_val is not None:
            score = accuracy_score(y_val, model.predict(X_val))
        else:
            score = accuracy_score(y_train, model.predict(X_train))
        if score > best_score:
            best_score = score
            best_model = model
        logging.info("Params %s -> accuracy %f", params, score)
    return best_model or train_model(X_train, y_train, log_to_mlflow=False)
