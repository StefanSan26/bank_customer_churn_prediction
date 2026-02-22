"""
FastAPI serving for Bank Customer Churn Prediction.
Health check and POST /predict using src.serving.inference.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.serving.inference import predict

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="ML API for predicting bank customer churn",
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check for load balancers and monitoring."""
    return {"status": "ok"}


class CustomerData(BaseModel):
    """Schema for one customer (features expected by the model)."""
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Surname: str


@app.post("/predict")
def get_prediction(data: CustomerData):
    """Return churn prediction (0 or 1) for the given customer. Input is raw (Geography, Gender, Surname); preprocessing is applied."""
    try:
        import pandas as pd
        from src.data.preprocess import preprocess_data
        from src.features.build_features import build_features
        df = pd.DataFrame([data.model_dump()])
        df = preprocess_data(df)
        X, _ = build_features(df)
        result = predict(features=X)
        out = result[0] if hasattr(result, "__getitem__") else result
        return {"prediction": int(out)}
    except Exception as e:
        return {"error": str(e)}
