from pathlib import Path

import joblib
import pandas as pd

from src.features import add_features

INPUT_PATH = Path("data/raw/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
OUTPUT_PATH = Path("data/processed/churn_predictions.csv")

def assign_risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= 0.15:
        return "Medium"
    return "Low"

def main() -> None:
    raw_data = pd.read_csv(INPUT_PATH)

    customer_ids = raw_data["customerID"].copy()

    features = raw_data.drop(columns=["customerID", "Churn"], errors="ignore")
    features["TotalCharges"] = pd.to_numeric(
        features["TotalCharges"],
        errors="coerce"
    )
    features = add_features(features)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "customerID": customer_ids,
        "churn_probability": probabilities.round(4),
        "risk_level": [
            assign_risk_level(probability, threshold)
            for probability in probabilities
        ],
        "predicted_churn": predictions
    })

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH.resolve()}")
    print(results.head(10))

if __name__ == "__main__":
    main()