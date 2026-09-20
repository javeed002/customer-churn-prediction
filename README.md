# Customer Churn Prediction — Web App

A complete, self-contained final-year project: train a churn prediction
model in Python, then serve it through a Flask website where you upload
customer data and get:

- An **overview dashboard** (churn rate, risk breakdown, top churn drivers)
- A **personalized dashboard for every customer** (churn probability, risk
  level, top risk/retention factors, suggested action)

A trained model is already included, so the site works immediately —
retraining is optional.

---

## 1. Project structure

```
churn_project/
├── README.md
├── requirements.txt
├── generate_sample_data.py     # creates the sample datasets (already run for you)
├── train_model.py              # trains the model (already run for you)
├── utils.py                    # shared data-cleaning/encoding functions
├── app.py                      # the Flask website
├── data/
│   ├── training_data.csv               # labeled data used to train the model
│   └── new_customers_to_predict.csv    # unlabeled sample data — upload this to try the site
├── model/
│   ├── churn_model.pkl          # trained RandomForest model
│   ├── feature_columns.json     # exact column order the model expects
│   └── feature_stats.json       # feature mean/std, used for per-customer explanations
├── templates/                   # HTML pages (Jinja2 + Bootstrap 5)
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   └── customer.html
├── static/
│   └── style.css
└── uploads/                     # uploaded CSV files land here
```

## 2. Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 3. Run the website

A trained model is already included in `model/`, so you can go straight to:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

1. On the home page, upload a CSV of customer data. To try it immediately,
   upload `data/new_customers_to_predict.csv` (already included).
2. You'll be redirected to the **overview dashboard**: total customers,
   predicted churn rate, risk-level breakdown chart, and the biggest
   churn drivers according to the model.
3. Click **View** next to any customer to open their **individual
   dashboard**: churn probability, risk level, their profile, the top
   factors pushing their risk up or down, and a suggested retention
   action.

Your uploaded file only needs the same columns as the training data
(gender, tenure, Contract, MonthlyCharges, etc.) — it does **not** need
a `Churn` column, since that's what gets predicted.

## 4. (Optional) Retrain the model

The included model was trained on the synthetic data in
`data/training_data.csv`. To retrain — for example after swapping in the
real [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
from Kaggle, or your own labeled data:

1. Replace `data/training_data.csv` with your own labeled CSV (must
   include a `Churn` column with Yes/No values, plus a `customerID`
   column).
2. Run:
   ```bash
   python train_model.py
   ```
   This prints accuracy/precision/recall/ROC-AUC on a held-out test set,
   then overwrites the files in `model/`.
3. Restart `app.py` to pick up the new model.

## 5. How the prediction & explanations work

- **Model**: a `RandomForestClassifier` (scikit-learn), trained with
  `class_weight="balanced"` to handle the fact that churners are usually
  a minority class.
- **Preprocessing**: categorical columns are one-hot encoded
  (`pd.get_dummies`); `utils.py` makes sure any newly uploaded data is
  aligned to the exact same columns the model was trained on, even if a
  category is missing or new.
- **Per-customer explanations**: for each customer, we combine the
  model's overall feature importances with how far that customer's
  values deviate from the training average (a z-score). Features that
  are both important *and* unusually high for that customer are shown
  as "risk factors"; features that are important but unusually low are
  shown as "retention factors."

  This is a lightweight, easy-to-explain heuristic — good for a student
  project and for giving a non-technical reader a clear "why." If you
  want a more rigorous, formally-grounded explanation method, the
  natural upgrade is **SHAP** (`shap.TreeExplainer(model)`), which
  computes exact per-prediction contribution values for tree-based
  models like this one.

## 6. Ideas for extending this project

- Swap in the real Telco Customer Churn dataset for training
- Add SHAP-based explanations instead of the importance × z-score heuristic
- Try other models (XGBoost, LightGBM, Logistic Regression) and compare
- Persist uploaded results in a real database (SQLite/PostgreSQL) instead
  of the in-memory `STATE` dict, so multiple users don't overwrite each
  other's data
- Deploy the site (Render, Railway, PythonAnywhere, or a VPS) so it's
  reachable outside your machine
- Add authentication if this were to handle real customer data
- Add a CSV export button on the dashboard for the predicted results

## 7. Notes & limitations

- This is a **single-user demo app**: uploading a new file overwrites
  the previously uploaded results (stored in memory, not a database).
  That's intentional for simplicity — see the extension ideas above if
  you want to support concurrent users.
- The included training data is **synthetic** (generated by
  `generate_sample_data.py`) to make the project runnable without any
  external downloads. For an official submission, retrain on the real
  Telco Customer Churn dataset for more meaningful accuracy numbers.
