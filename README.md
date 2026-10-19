# Customer Churn Predictor

An end-to-end machine learning project for identifying telecom customers at risk of churn.

## Business Goal

The goal is to identify customers who are likely to leave the service, allowing the business to take proactive retention actions such as targeted offers, contract upgrades, or support outreach.

## Dataset

The project uses the Telco Customer Churn dataset:

- 7,043 customers
- 21 original features
- Binary target: `Churn`
- Churn rate: 26.5%

## Exploratory Data Analysis

Key findings:

- Customers with month-to-month contracts have the highest churn rate.
- Churn is highest among customers with 0–6 months of tenure.
- Customers without Online Security or Tech Support have higher churn risk.
- Fiber optic customers are strongly associated with a higher churn risk.
- Two-year contracts and longer tenure are associated with lower churn risk.

## Modeling

The following classification models were compared:

- Logistic Regression
- K-Nearest Neighbors
- Linear SVM
- Kernel SVM
- Gaussian Naive Bayes
- Decision Tree
- Random Forest

Logistic Regression was selected as the final model because it achieved the best PR-AUC on the validation set.

A classification threshold of `0.30` was selected to prioritize recall and identify as many potential churners as possible.

## Final Test Results

| Metric | Score |
|---|---:|
| Recall | 0.9305 |
| Precision | 0.4345 |
| F1 Score | 0.5923 |
| PR-AUC | 0.6666 |
| Classification Threshold | 0.30 |

The model identified 348 of 374 churned customers in the held-out test set.

## Features

- Data analysis notebooks with visualizations and business insights
- Model comparison and threshold selection
- Model interpretability using Logistic Regression coefficients
- Batch CSV prediction script
- FastAPI endpoint for single-customer predictions
- Streamlit web interface for individual and batch predictions
- Downloadable CSV file with churn probabilities and risk levels

## Project Structure

```text
customer-churn-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_model_comparison.ipynb
├── src/
│   ├── api.py
│   ├── data.py
│   ├── features.py
│   ├── predict.py
│   └── train_baseline.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
uv venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

## Run Batch Predictions

```bash
python -m src.predict
```

The output file is saved to:

```text
data/processed/churn_predictions.csv
```

## Run the API

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run the Web Application

```bash
streamlit run app.py
```

Open the address shown in the terminal, usually:

```text
http://localhost:8501
```

## Business Recommendation

Prioritize proactive retention efforts for customers with:

- Month-to-month contracts
- Fiber optic internet service
- Short tenure
- No Online Security or Tech Support
- Electronic check payment method

Model results represent associations in the available dataset and should support, not replace, business decision-making.