from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import add_features

MODEL_PATH = Path("models/churn_model.joblib")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]

st.title("Customer Churn Predictor")
st.caption("Estimate churn risk for an individual customer.")

with st.form("customer_form"):
    left_column, middle_column, right_column = st.columns(3)

    with left_column:
        customer_id = st.text_input("Customer ID", value="demo-001")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.checkbox("Senior Citizen")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

    with middle_column:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        if phone_service == "No":
            multiple_lines = "No phone service"
            st.text_input("Multiple Lines", value=multiple_lines, disabled=True)
        else:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with right_column:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=29.85,
            step=0.01
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=29.85,
            step=0.01
        )

    submitted = st.form_submit_button("Predict Churn Risk", type="primary")

if submitted:
    customer = pd.DataFrame([{
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    features = customer.drop(columns=["customerID"])
    features = add_features(features)

    probability = float(model.predict_proba(features)[0, 1])
    predicted_churn = probability >= threshold

    if predicted_churn:
        risk_level = "High"
        st.error("High churn risk")
    elif probability >= 0.15:
        risk_level = "Medium"
        st.warning("Medium churn risk")
    else:
        risk_level = "Low"
        st.success("Low churn risk")

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Churn Probability", f"{probability:.1%}")
    metric_two.metric("Risk Level", risk_level)
    metric_three.metric("Decision Threshold", f"{threshold:.0%}")

    st.progress(int(probability * 100))

    if predicted_churn:
        st.info(
            "Recommend proactive retention outreach. "
            "Consider a contract upgrade or support-related offer."
        )
st.divider()
st.subheader("Batch Prediction")
st.caption("Upload a CSV file to score multiple customers at once.")

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type="csv"
)

required_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    missing_columns = [
        column
        for column in required_columns
        if column not in batch_data.columns
    ]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        customer_ids = batch_data.get(
            "customerID",
            pd.Series(range(1, len(batch_data) + 1))
        )

        batch_features = batch_data.drop(
            columns=["customerID", "Churn"],
            errors="ignore"
        )

        batch_features["TotalCharges"] = pd.to_numeric(
            batch_features["TotalCharges"],
            errors="coerce"
        )

        batch_features = add_features(batch_features)

        probabilities = model.predict_proba(batch_features)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        risk_levels = [
            "High" if probability >= threshold
            else "Medium" if probability >= 0.15
            else "Low"
            for probability in probabilities
        ]

        batch_results = pd.DataFrame({
            "customerID": customer_ids,
            "churn_probability": probabilities.round(4),
            "risk_level": risk_levels,
            "predicted_churn": predictions
        })

        high_risk_count = int((batch_results["risk_level"] == "High").sum())

        first_metric, second_metric = st.columns(2)
        first_metric.metric("Customers Processed", len(batch_results))
        second_metric.metric("High-Risk Customers", high_risk_count)

        st.dataframe(batch_results.head(100), use_container_width=True)

        st.download_button(
            label="Download Predictions CSV",
            data=batch_results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )