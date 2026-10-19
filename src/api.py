from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features import add_features

MODEL_PATH = Path("models/churn_model.joblib")

app = FastAPI(
    title="Customer Churn Predictor API",
    version="1.0.0"
)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict")
def predict_churn(customer: dict[str, Any]) -> dict[str, Any]:
    try:
        customer_id = customer.get("customerID")

        data = pd.DataFrame([customer])
        features = data.drop(columns=["customerID", "Churn"], errors="ignore")

        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"],
            errors="coerce"
        )

        features = add_features(features)

        probability = float(model.predict_proba(features)[0, 1])
        predicted_churn = int(probability >= threshold)

        if probability >= threshold:
            risk_level = "High"
        elif probability >= 0.15:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "customerID": customer_id,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "predicted_churn": predicted_churn,
            "threshold": threshold
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error