"""Build feature matrix and target from preprocessed bank churn data."""
import numpy as np
import pandas as pd

# Columns to drop from features (IDs and target)
ID_AND_TARGET = {"CustomerId", "id", "Exited"}


def _add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add interaction, ratio, bucket, and frequency features."""
    X = X.copy()

    if "NumOfProducts" in X.columns and "IsActiveMember" in X.columns:
        X["Mem__no__Products"] = X["NumOfProducts"] * X["IsActiveMember"]
    if "Balance" in X.columns and "EstimatedSalary" in X.columns:
        X["Balance_Salary_Ratio"] = np.where(
            X["EstimatedSalary"] > 0, X["Balance"] / X["EstimatedSalary"], 0.0,
        )
    if "Balance" in X.columns and "Age" in X.columns:
        X["Balance_Age_Ratio"] = np.where(X["Age"] > 0, X["Balance"] / X["Age"], 0.0)

    if "Age" in X.columns:
        X["AgeBucket"] = pd.cut(
            X["Age"], bins=[0, 25, 35, 45, 55, 120], labels=[0, 1, 2, 3, 4], include_lowest=True,
        ).astype(int)
    if "Tenure" in X.columns:
        X["TenureBucket"] = pd.cut(
            X["Tenure"], bins=[0, 2, 5, 10, 20], labels=[0, 1, 2, 3], include_lowest=True,
        ).astype(int)

    if "Age" in X.columns and "Tenure" in X.columns:
        X["Age_Tenure"] = X["Age"] * X["Tenure"]
    if "CreditScore" in X.columns and "IsActiveMember" in X.columns:
        X["CreditScore_IsActive"] = X["CreditScore"] * X["IsActiveMember"]
    if "Tenure" in X.columns and "NumOfProducts" in X.columns:
        X["Tenure_NumProducts"] = X["Tenure"] * X["NumOfProducts"]

    if "Geography" in X.columns:
        geo_freq = X["Geography"].value_counts(normalize=True)
        X["Geography_freq"] = X["Geography"].map(geo_freq)

    return X


def build_features(df: pd.DataFrame):
    """
    Build X (features) and y (target) from preprocessed DataFrame.

    Drops Exited, CustomerId, id if present, then adds engineered features
    (buckets, ratios, interactions, frequency encoding). Returns (X, y) where
    y is Exited.

    Args:
        df: Preprocessed DataFrame with Exited column.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series or None).
    """
    drop_cols = [c for c in ID_AND_TARGET if c in df.columns]
    X = df.drop(columns=drop_cols)
    if "Exited" in X.columns:
        X = X.drop(columns=["Exited"])

    X = _add_engineered_features(X)

    y = df["Exited"] if "Exited" in df.columns else None
    return X, y
