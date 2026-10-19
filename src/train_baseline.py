"""Train reproducible churn-model baselines without preprocessing leakage."""

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_raw_data, split_features_target


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def metrics(y_true, probabilities, predictions) -> dict[str, float]:
    return {
        "recall": round(float(recall_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
    }


def main() -> None:
    df = load_raw_data(DATA_PATH)
    X, y = split_features_target(df)

    # First split: reserve a final test set. Second split: validation for model choices.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([( "imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    logistic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)

    results = {}
    for name, model in {"dummy": dummy, "logistic_regression": logistic_pipeline}.items():
        probabilities = model.predict_proba(X_val)[:, 1]
        results[name] = metrics(y_val, probabilities, (probabilities >= 0.5).astype(int))

    results["splits"] = {"train": len(X_train), "validation": len(X_val), "test_held_out": len(X_test)}
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "baseline_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(logistic_pipeline, MODELS_DIR / "logistic_regression_pipeline.joblib")
    print(json.dumps(results, indent=2))
    print("Saved model pipeline and validation metrics in models/.")


if __name__ == "__main__":
    main()
