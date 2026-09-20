"""
Generates synthetic customer data (styled after the Telco Customer Churn
dataset) so the project works out of the box without needing you to
download anything first.

Produces:
  data/training_data.csv            -> labeled data (has Churn column), used by train_model.py
  data/new_customers_to_predict.csv -> unlabeled data (no Churn column), upload this on the website to try it out
"""
import numpy as np
import pandas as pd

np.random.seed(42)


def make_customers(n, start_id=1):
    ids = [f"CUST{str(i).zfill(5)}" for i in range(start_id, start_id + n)]
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.85, 0.15])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = np.random.randint(0, 73, n)
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], n)
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    online_security = np.random.choice(["Yes", "No", "No internet service"], n)
    tech_support = np.random.choice(["Yes", "No", "No internet service"], n)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    monthly_charges = np.round(np.random.uniform(18.0, 120.0, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(0, 50, n), 2)

    df = pd.DataFrame({
        "customerID": ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    })
    return df


def add_churn_label(df):
    churn_prob = (
        0.05
        + 0.35 * (df["Contract"] == "Month-to-month")
        + 0.15 * (df["tenure"] < 12)
        + 0.10 * (df["MonthlyCharges"] > 80)
        + 0.08 * (df["TechSupport"] == "No")
    )
    churn_prob = np.clip(churn_prob, 0, 0.9)
    df = df.copy()
    df["Churn"] = ["Yes" if np.random.rand() < p else "No" for p in churn_prob]
    return df


if __name__ == "__main__":
    # Training set (labeled)
    train_df = make_customers(500, start_id=1)
    train_df = add_churn_label(train_df)
    train_df.to_csv("data/training_data.csv", index=False)
    print("Saved data/training_data.csv ->", train_df.shape)

    # New, unlabeled customers to upload on the website and get predictions for
    new_df = make_customers(20, start_id=501)
    new_df.to_csv("data/new_customers_to_predict.csv", index=False)
    print("Saved data/new_customers_to_predict.csv ->", new_df.shape)
