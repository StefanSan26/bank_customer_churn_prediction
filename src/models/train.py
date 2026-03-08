"""Train CatBoost classifier for bank churn prediction."""
import logging
from typing import Any, Optional

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature

DEFAULT_PARAMS = {
    "subsample": 0.8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 1,
    "depth": 4,
    "random_seed": 42,
}


def _log_loss_curve(evals_result: dict, log_every_n: int = 10) -> None:
    """Log per-iteration loss metrics from CatBoost eval results to MLflow."""
    for dataset_name, metric_dict in evals_result.items():
        tag = "train" if dataset_name == "learn" else "val"
        for metric_name, values in metric_dict.items():
            mlflow_key = f"{tag}_{metric_name}"
            for step, value in enumerate(values):
                if step % log_every_n == 0 or step == len(values) - 1:
                    mlflow.log_metric(mlflow_key, value, step=step)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[dict[str, Any]] = None,
    log_to_mlflow: bool = True,
    verbose: int = 0,
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier and optionally log to MLflow.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Optional validation features for loss curve tracking.
        y_val: Optional validation target for loss curve tracking.
        params: CatBoost hyperparameters (defaults used if None).
        log_to_mlflow: If True, log params, metrics, and model to the active MLflow run.
        verbose: CatBoost verbosity (0 = silent).

    Returns:
        Fitted CatBoostClassifier.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = CatBoostClassifier(**p)

    fit_kwargs: dict[str, Any] = {"verbose": verbose}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = (X_val, y_val)

    model.fit(X_train, y_train, **fit_kwargs)

    if log_to_mlflow:
        mlflow.log_params(p)

        evals_result = model.get_evals_result()
        if evals_result:
            _log_loss_curve(evals_result)

        try:
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.catboost.log_model(
                model,
                "model",
                signature=signature,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to log CatBoost model to MLflow: %s", e)

    return model
