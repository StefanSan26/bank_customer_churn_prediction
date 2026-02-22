"""Data validation for bank churn pipeline."""
import logging
from typing import List, Tuple

import pandas as pd

REQUIRED_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Surname", "Exited",
]


def validate_bank_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required columns and non-empty. No Great Expectations.

    Args:
        df: Raw or preprocessed DataFrame.

    Returns:
        (is_valid, list of failure messages).
    """
    failed: List[str] = []
    if df is None or df.empty:
        failed.append("DataFrame is None or empty")
        return False, failed
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        failed.append(f"Missing columns: {missing}")
    return len(failed) == 0, failed
