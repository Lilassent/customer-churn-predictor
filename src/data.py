"""Dataset loading and cleaning functions."""

from pathlib import Path

import pandas as pd


TARGET = "Churn"
ID_COLUMN = "customerID"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the Telco churn CSV and apply deterministic cleaning."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    if df[TARGET].isna().any():
        raise ValueError("Target column contains values other than 'Yes' or 'No'.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and the binary churn target."""
    features = df.drop(columns=[TARGET, ID_COLUMN])
    return features, df[TARGET]
