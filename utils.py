"""
Shared preprocessing helpers.
Used by BOTH train_model.py (training time) and app.py (prediction time)
so that new customer data is transformed exactly the same way the
training data was.
"""

import pandas as pd


def _is_text_dtype(series):
    """True for classic object-dtype strings AND pandas' newer StringDtype
    (pandas >= 2.x/3.x can default to the latter)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def clean_data(df):
    """Basic cleaning: fix TotalCharges type, fill missing values, cast flags."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Drop fully empty rows, if any slipped in
    df = df.dropna(how="all")

    return df


def encode_features(df, drop_first=True):
    """One-hot encode all categorical (object) columns except customerID."""
    df = df.copy()
    cat_cols = [c for c in df.columns if _is_text_dtype(df[c])]
    cat_cols = [c for c in cat_cols if c != "customerID"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df_encoded


def align_columns(df_encoded, feature_columns):
    """
    Make sure a (new/uploaded) encoded dataframe has EXACTLY the same
    columns, in the same order, as the data the model was trained on.
    - Missing columns (categories not seen in the new data) -> filled with 0
    - Extra columns (categories not seen during training) -> dropped
    """
    df_encoded = df_encoded.copy()

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[feature_columns]


def risk_level(prob):
    """Turn a churn probability into a simple Low/Medium/High label."""
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


def encode_target(df):
    """Map a Yes/No Churn column to 1/0, whatever its exact string dtype."""
    df = df.copy()
    if "Churn" in df.columns and _is_text_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def prettify_column(col):
    """Turn a one-hot encoded column name like 'Contract_Month-to-month'
    into a readable label like 'Contract: Month-to-month'."""
    if "_" in col:
        parts = col.split("_", 1)
        return f"{parts[0]}: {parts[1]}"
    return col
