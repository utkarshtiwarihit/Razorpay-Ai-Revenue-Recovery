"""
02_preprocessing_training
--------------------------
Loads data/customers.csv, preprocesses it, trains a Random Forest
classifier to predict risk_category, evaluates it, calibrates a
0-100 risk score, computes feature importance, and saves everything
the Streamlit app needs.

Run:  python notebooks/train_model.py
Output:
  models/risk_model.pkl
  models/preprocessing.pkl
  models/feature_importance.csv
  data/customers_scored.csv   (customers.csv + predicted risk_score / risk_category_pred)
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
df = pd.read_csv(os.path.join(data_dir, "customers.csv"))
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ----------------------------------------------------------------------
# 2. Basic cleaning
# ----------------------------------------------------------------------
print("Duplicates:", df.duplicated(subset="customer_id").sum())
df = df.drop_duplicates(subset="customer_id").reset_index(drop=True)

print("Missing values before fix:\n", df.isna().sum()[df.isna().sum() > 0])
# engagement_score had a few NaNs by design -> fill with median (simple, safe for demo)
df["engagement_score"] = df["engagement_score"].fillna(df["engagement_score"].median())

# ----------------------------------------------------------------------
# 3. Feature selection
#    We deliberately EXCLUDE identifying columns (customer_id, email,
#    phone) and the "raw" latent indicator (that would be data leakage -
#    it was used to build the label) from the model's inputs.
# ----------------------------------------------------------------------
categorical_features = ["employment_type", "payment_frequency", "previous_reminder_response"]
numeric_features = [
    "age", "monthly_income", "account_age_months", "credit_score",
    "loan_amount", "monthly_installment", "previous_late_payments",
    "avg_payment_delay_days", "previous_defaults", "days_past_due",
    "current_outstanding", "payment_reliability_score", "engagement_score",
    "amount_due",
]

X = df[categorical_features + numeric_features].copy()
y = df["risk_category"].copy()

# ----------------------------------------------------------------------
# 4. Encode categoricals (simple LabelEncoder per column - fine for
#    tree-based models and easy for beginners to understand)
# ----------------------------------------------------------------------
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ----------------------------------------------------------------------
# 5. Train/test split BEFORE scaling to avoid data leakage
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 6. Scale numeric features (fit ONLY on train, then apply to test)
#    Random Forest doesn't strictly need scaling, but we keep the
#    pipeline realistic + reusable if we ever swap in Logistic Regression.
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# ----------------------------------------------------------------------
# 7. Train model
# ----------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",   # important: CRITICAL is the rarest class
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 8. Evaluate
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}  <-- most important: missing a real CRITICAL customer is costly")
print(f"F1-score : {f1:.3f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
labels_order = ["LOW", "AT-RISK", "HIGH", "CRITICAL"]
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
print(pd.DataFrame(cm, index=labels_order, columns=labels_order))
print("\nFull classification report:")
print(classification_report(y_test, y_pred, labels=labels_order, zero_division=0))

# ----------------------------------------------------------------------
# 9. Feature importance
# ----------------------------------------------------------------------
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
importance_df.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)
print("\nTop features:\n", importance_df.head(8))

# ----------------------------------------------------------------------
# 10. Turn probabilities into a calibrated 0-100 risk score.
#     Rather than using P(class) directly (which jumps abruptly between
#     classes), we build a weighted "severity score":
#     LOW=0, AT-RISK=1, HIGH=2, CRITICAL=3 weighted by each class's
#     predicted probability -> smooth 0-100 scale that still lines up
#     with the class thresholds.
# ----------------------------------------------------------------------
severity_weight = {"LOW": 0, "AT-RISK": 1, "HIGH": 2, "CRITICAL": 3}
class_order = list(model.classes_)  # order sklearn uses internally

def probs_to_score(proba_row):
    weighted = sum(p * severity_weight[c] for p, c in zip(proba_row, class_order))
    return round((weighted / 3) * 100, 1)  # normalize 0-3 -> 0-100

# Score the FULL dataset (not just test set) for the dashboard demo
X_full = df[categorical_features + numeric_features].copy()
for col in categorical_features:
    X_full[col] = encoders[col].transform(X_full[col])
X_full[numeric_features] = scaler.transform(X_full[numeric_features])

full_proba = model.predict_proba(X_full)
full_pred = model.predict(X_full)
full_scores = np.array([probs_to_score(row) for row in full_proba])

scored = df.copy()
scored["risk_category_pred"] = full_pred
scored["risk_score"] = full_scores

scored.to_csv(os.path.join(data_dir, "customers_scored.csv"), index=False)
print(f"\nSaved scored dataset -> data/customers_scored.csv")

# ----------------------------------------------------------------------
# 11. Save model + preprocessing objects
# ----------------------------------------------------------------------
joblib.dump(model, os.path.join(models_dir, "risk_model.pkl"))
joblib.dump({
    "encoders": encoders,
    "scaler": scaler,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "class_order": class_order,
    "severity_weight": severity_weight,
}, os.path.join(models_dir, "preprocessing.pkl"))

print("Saved models/risk_model.pkl and models/preprocessing.pkl")
print("\nDONE.")
