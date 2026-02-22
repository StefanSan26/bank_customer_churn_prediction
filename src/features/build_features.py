"""Build feature matrix and target from preprocessed bank churn data."""
import pandas as pd

# Columns to drop from features (IDs and target)
ID_AND_TARGET = {"CustomerId", "id", "Exited"}


def build_features(df: pd.DataFrame):
    """
    Build X (features) and y (target) from preprocessed DataFrame.

    Drops Exited, CustomerId, id if present. Returns (X, y) where y is Exited.

    Args:
        df: Preprocessed DataFrame with Exited column.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series).
    """
    drop_cols = [c for c in ID_AND_TARGET if c in df.columns]
    X = df.drop(columns=drop_cols)
    if "Exited" in df.columns:
        y = df["Exited"]
        if "Exited" in X.columns:
            X = X.drop(columns=["Exited"])
        return X, y
    return X, None
