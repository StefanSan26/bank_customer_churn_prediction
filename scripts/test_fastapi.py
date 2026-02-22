#!/usr/bin/env python3
"""Smoke test for FastAPI: GET / and POST /predict. Start server first: uvicorn src.app.main:app --reload."""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def main():
    print("=== FastAPI smoke test ===")
    r = requests.get(f"{BASE}/", timeout=5)
    assert r.status_code == 200, r.text
    print("GET / OK:", r.json())

    payload = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 100000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 150000.0,
        "Surname": "Smith",
    }
    r = requests.post(f"{BASE}/predict", json=payload, timeout=10)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "prediction" in out or "error" in out
    print("POST /predict OK:", out)
    print("FastAPI test OK.")


if __name__ == "__main__":
    main()
