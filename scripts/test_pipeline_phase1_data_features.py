#!/usr/bin/env python3
"""Smoke test: load -> preprocess -> build_features (no training)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# Use data/train.csv or data/raw if present
DATA_PATH = "data/train.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data/raw/train.csv"


def main():
    print("=== Phase 1: Load -> Preprocess -> Build Features ===")
    if not os.path.exists(DATA_PATH):
        print(f"Skip: {DATA_PATH} not found. Create it or put raw CSV in data/raw/.")
        return
    print("[1] Loading...")
    df = load_data(DATA_PATH)
    print(f"    Shape: {df.shape}")
    print("[2] Preprocessing...")
    df = preprocess_data(df)
    print(f"    Shape: {df.shape}")
    print("[3] Building features...")
    X, y = build_features(df)
    print(f"    X: {X.shape}, y: {y.shape if y is not None else None}")
    print("Phase 1 OK.")


if __name__ == "__main__":
    main()
