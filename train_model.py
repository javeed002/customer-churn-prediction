"""
Trains the churn prediction model on data/training_data.csv and saves:
  model/churn_model.pkl        -> the trained model
  model/feature_columns.json   -> exact column order the model expects
  model/feature_stats.json     -> mean/std of each feature (used for the
                                  per-customer "risk factor" explanations
                                  shown in the web app)

Run this whenever you want to retrain on new / updated data:
    python train_model.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split

from utils import clean_data, encode_features, encode_target

DATA_PATH = "data/training_data.csv"
MODEL_DIR = "model"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading training data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    df = encode_target(df)

    df_model = df.drop(columns=["customerID"], errors="ignore")
    df_encoded = encode_features(df_model)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight="balanced" handles churn-class imbalance without needing
    # an extra library (SMOTE is a good upgrade if you want to add it later)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation on held-out test data ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Save the model
    joblib.dump(model, f"{MODEL_DIR}/churn_model.pkl")

    # Save the exact feature/column order the model expects, so the web app
    # can align any newly uploaded data to match it
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    # Save mean/std of the training features, used to build simple,
    # human-readable "why is this customer at risk" explanations
    stats = {
        "means": X.mean().to_dict(),
        "stds": X.std().replace(0, 1).to_dict(),
    }
    with open(f"{MODEL_DIR}/feature_stats.json", "w") as f:
        json.dump(stats, f)

    print(f"\nSaved model + metadata to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
