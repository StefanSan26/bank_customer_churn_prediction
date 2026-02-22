#!/usr/bin/env python3
"""Build processed dataset from raw: load -> preprocess -> save to data/processed/."""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

# Default paths: raw or fallback to data/train.csv
RAW = os.environ.get("BANK_RAW_DATA", "data/train.csv")
OUT = os.path.join(ROOT, "data", "processed", "bank_churn_processed.csv")


def main():
    df = load_data(RAW)
    df = preprocess_data(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Processed dataset saved to {OUT} | Shape: {df.shape}")


if __name__ == "__main__":
    main()
