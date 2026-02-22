"""Preprocess bank churn data: encoding, cleaning, surname hashing."""
import hashlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _hash_surname(surname) -> int:
    """Hash surname to a consistent integer (0-999)."""
    hash_obj = hashlib.md5(str(surname).encode())
    return int(hash_obj.hexdigest()[:8], 16) % 1000


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw bank churn DataFrame: dropna, label-encode Gender/Geography, hash Surname.

    Args:
        df: Raw DataFrame with columns including Gender, Geography, Surname, Exited.

    Returns:
        Preprocessed DataFrame (same columns, encoded).
    """
    data = df.dropna().reset_index(drop=True)
    label_enc_gender = LabelEncoder()
    label_enc_geography = LabelEncoder()
    data = data.copy()
    data["Gender"] = label_enc_gender.fit_transform(data["Gender"])
    data["Geography"] = label_enc_geography.fit_transform(data["Geography"])
    data["Surname"] = data["Surname"].apply(_hash_surname)
    return data
