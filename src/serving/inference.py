"""Load model and run predictions for bank churn."""
import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd

# Optional MLflow; if not available we can load from local path
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def load_model(model_uri: Optional[str] = None):
    """
    Load CatBoost model from MLflow (Model Registry or run URI) or from local path.

    Args:
        model_uri: e.g. "models:/bank_churn_prediction/Staging" or "runs:/<run_id>/model".
                   If None, uses MLflow registry "models:/bank_churn_prediction/Staging".

    Returns:
        Loaded model (e.g. CatBoostClassifier).
    """
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("mlflow is required for load_model")
    uri = model_uri or "models:/bank_churn_prediction/Staging"
    return mlflow.catboost.load_model(uri)


def predict(
    model_or_uri: Optional[Union[Any, str]] = None,
    features: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
) -> Union[list, pd.Series]:
    """
    Run prediction (and optionally predict_proba) for bank churn.

    Args:
        model_or_uri: Fitted model or MLflow model URI. If None, loads from registry.
        features: DataFrame or dict of features (one row). If None, returns empty list.

    Returns:
        Predictions (class labels). If features is a DataFrame, returns a Series/array.
    """
    if features is None:
        return []
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    if model_or_uri is None:
        model = load_model()
    elif isinstance(model_or_uri, str):
        model = load_model(model_or_uri)
    else:
        model = model_or_uri
    return model.predict(features)
