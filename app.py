"""
Customer Churn Prediction Website
---------------------------------
Flask app that:
  1. Lets you upload a CSV of customer data (no Churn column needed)
  2. Predicts churn probability for every customer using the trained model
  3. Shows an overview dashboard (churn rate, risk breakdown, top drivers)
  4. Shows an individual dashboard per customer with a plain-English
     explanation of why they're flagged at risk

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from utils import align_columns, clean_data, encode_features, prettify_column, risk_level

app = Flask(__name__)
app.secret_key = "churn-dashboard-demo-secret"  # fine for a local demo project

MODEL_PATH = "model/churn_model.pkl"
COLUMNS_PATH = "model/feature_columns.json"
STATS_PATH = "model/feature_stats.json"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
    raise RuntimeError(
        "No trained model found. Run 'python train_model.py' first "
        "to create model/churn_model.pkl."
    )

model = joblib.load(MODEL_PATH)

with open(COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

with open(STATS_PATH) as f:
    FEATURE_STATS = json.load(f)

FEATURE_MEANS = pd.Series(FEATURE_STATS["means"])
FEATURE_STDS = pd.Series(FEATURE_STATS["stds"])
FEATURE_IMPORTANCES = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)

# Simple in-memory store for the currently uploaded/predicted dataset.
# This is a single-user demo app, not built for concurrent multi-user
# production use - for that you'd move this into a real database.
STATE = {"df": None, "aligned": None}


def run_predictions(df_raw):
    """Clean, encode, align and predict churn for an uploaded customer file."""
    df = clean_data(df_raw)

    has_id = "customerID" in df.columns
    if has_id:
        customer_ids = df["customerID"].astype(str)
    else:
        customer_ids = pd.Series([f"CUST{i + 1:04d}" for i in range(len(df))])

    df_for_model = df.drop(columns=["customerID", "Churn"], errors="ignore")
    df_encoded = encode_features(df_for_model)
    df_aligned = align_columns(df_encoded, FEATURE_COLUMNS)

    probs = model.predict_proba(df_aligned)[:, 1]
    preds = model.predict(df_aligned)

    result = df.drop(columns=["Churn"], errors="ignore").copy()
    result["customerID"] = customer_ids.values
    result["ChurnProbability"] = probs.round(3)
    result["ChurnPrediction"] = ["Yes" if p == 1 else "No" for p in preds]
    result["RiskLevel"] = [risk_level(p) for p in probs]

    df_aligned = df_aligned.reset_index(drop=True)
    result = result.reset_index(drop=True)

    return result, df_aligned


def get_reasons(row_aligned, top_n=4):
    """
    Build a simple, human-readable explanation for one customer's
    prediction: which of their features are (a) more important to the
    model AND (b) higher than the typical customer, driving risk up
    (or down).

    This is a lightweight heuristic (importance x z-score), not a formal
    method like SHAP - good enough to explain a prediction to a
    non-technical reader. Swap in SHAP's TreeExplainer for a more
    rigorous version if you extend this project.
    """
    row = row_aligned.iloc[0]
    z_scores = (row - FEATURE_MEANS) / FEATURE_STDS
    contributions = (FEATURE_IMPORTANCES * z_scores).sort_values(ascending=False)

    risk_factors = contributions[contributions > 0].head(top_n)
    protective_factors = contributions[contributions < 0].sort_values().head(top_n)

    risk_factors = {prettify_column(k): round(v, 3) for k, v in risk_factors.items()}
    protective_factors = {prettify_column(k): round(v, 3) for k, v in protective_factors.items()}

    return risk_factors, protective_factors


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please choose a CSV file to upload.")
            return redirect(url_for("index"))
        if not file.filename.lower().endswith(".csv"):
            flash("Please upload a .csv file.")
            return redirect(url_for("index"))

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            df_raw = pd.read_csv(filepath)
            result, aligned = run_predictions(df_raw)
        except Exception as exc:  # surface a friendly error instead of a 500 page
            flash(f"Couldn't process that file: {exc}")
            return redirect(url_for("index"))

        STATE["df"] = result
        STATE["aligned"] = aligned

        return redirect(url_for("dashboard"))

    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    df = STATE.get("df")
    if df is None:
        flash("Upload a customer data file first.")
        return redirect(url_for("index"))

    total = len(df)
    churn_count = int((df["ChurnPrediction"] == "Yes").sum())
    churn_rate = round(churn_count / total * 100, 1) if total else 0

    risk_counts = df["RiskLevel"].value_counts()
    risk_labels = ["High", "Medium", "Low"]
    risk_values = [int(risk_counts.get(r, 0)) for r in risk_labels]

    top_features = FEATURE_IMPORTANCES.sort_values(ascending=False).head(8)
    top_feature_labels = [prettify_column(c) for c in top_features.index]
    top_feature_values = [round(v, 4) for v in top_features.values]

    customers = df.sort_values("ChurnProbability", ascending=False).to_dict(orient="records")

    return render_template(
        "dashboard.html",
        total=total,
        churn_count=churn_count,
        churn_rate=churn_rate,
        risk_labels=risk_labels,
        risk_values=risk_values,
        top_feature_labels=top_feature_labels,
        top_feature_values=top_feature_values,
        customers=customers,
    )


@app.route("/customer/<customer_id>")
def customer_detail(customer_id):
    df = STATE.get("df")
    aligned = STATE.get("aligned")
    if df is None:
        flash("Upload a customer data file first.")
        return redirect(url_for("index"))

    match = df[df["customerID"] == customer_id]
    if match.empty:
        flash("Customer not found.")
        return redirect(url_for("dashboard"))

    idx = match.index[0]
    row = match.iloc[0]
    row_aligned = aligned.loc[[idx]]

    risk_factors, protective_factors = get_reasons(row_aligned)

    return render_template(
        "customer.html",
        customer=row.to_dict(),
        risk_factors=risk_factors,
        protective_factors=protective_factors,
    )


if __name__ == "__main__":
    app.run(debug=True)
