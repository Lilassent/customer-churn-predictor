import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

TENURE_BINS = [-1, 6, 12, 24, 48, float("inf")]

TENURE_LABELS = [
    "0-6",
    "7-12",
    "13-24",
    "25-48",
    "49+"
]

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["ServiceCount"] = result[SERVICE_COLUMNS].eq("Yes").sum(axis=1)

    result["TenureGroup"] = pd.cut(
        result["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS
    )

    result["MonthlyChargesPerTenure"] = np.where(
        result["tenure"] > 0,
        result["MonthlyCharges"] / result["tenure"],
        0
    )

    return result