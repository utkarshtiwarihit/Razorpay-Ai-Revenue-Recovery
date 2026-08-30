"""
01_data_generation
-------------------
Generates a synthetic (100% fake) dataset of 1,000 customers for the
Razorpay AI Finance Risk Engine hackathon demo.

NO real names / emails / phone numbers / PAN / Aadhaar are used anywhere.
Everything is clearly synthetic: CUSTOMER_0001, customer0001@example.com, etc.

Run:  python notebooks/generate_data.py
Output: data/customers.csv, data/recovery_data.csv
"""

import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------
# 1. Reproducibility
# ----------------------------------------------------------------------
np.random.seed(42)
N = 1000

# ----------------------------------------------------------------------
# 2. Create a hidden "true risk" latent variable (0 = perfectly safe,
#    1 = extremely risky). Every visible column will be nudged by this
#    latent value + random noise, so relationships feel realistic but
#    are not perfectly obvious to the model.
# ----------------------------------------------------------------------
latent_risk = np.random.beta(a=2, b=5, size=N)  # skewed toward "safer" customers

customer_id = [f"CUSTOMER_{i+1:04d}" for i in range(N)]
email = [f"customer{i+1:04d}@example.com" for i in range(N)]
# Clearly synthetic demo phone numbers (not real, using a fake prefix block)
phone = [f"+91-9000{np.random.randint(0,10):01d}{i:05d}" for i in range(N)]

# ----------------------------------------------------------------------
# 3. Demographics & employment
# ----------------------------------------------------------------------
age = np.clip(np.random.normal(35, 9, N), 21, 65).astype(int)

employment_type = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Freelancer"],
    size=N, p=[0.55, 0.20, 0.15, 0.10]
)

# Self-employed / freelancers have slightly higher income volatility -> nudge risk a bit
employment_risk_bump = np.where(
    np.isin(employment_type, ["Freelancer", "Self-Employed"]), 0.05, 0.0
)
latent_risk_adj = np.clip(latent_risk + employment_risk_bump, 0, 1)

monthly_income = np.clip(
    np.random.normal(55000, 22000, N) * (1 - 0.5 * latent_risk_adj), 12000, 300000
).astype(int)

account_age_months = np.clip(
    np.random.normal(30, 15, N) * (1 - 0.3 * latent_risk_adj), 1, 120
).astype(int)

# ----------------------------------------------------------------------
# 4. Credit & loan profile
# ----------------------------------------------------------------------
credit_score = np.clip(
    (780 - 380 * latent_risk_adj) + np.random.normal(0, 35, N), 300, 900
).astype(int)

loan_amount = np.clip(
    np.random.normal(180000, 90000, N) * (0.7 + 0.6 * latent_risk_adj), 20000, 800000
).astype(int)

monthly_installment = np.clip(
    (loan_amount / np.random.uniform(6, 36, N)), 1000, 60000
).astype(int)

# ----------------------------------------------------------------------
# 5. Payment behaviour history
# ----------------------------------------------------------------------
previous_late_payments = np.clip(
    np.random.poisson(lam=1 + 8 * latent_risk_adj, size=N), 0, 24
).astype(int)

avg_payment_delay_days = np.clip(
    np.random.exponential(scale=2 + 15 * latent_risk_adj, size=N), 0, 90
).astype(int)

previous_defaults = np.clip(
    np.random.poisson(lam=0.2 + 1.5 * latent_risk_adj, size=N), 0, 6
).astype(int)

days_past_due = np.clip(
    np.random.exponential(scale=1 + 40 * latent_risk_adj, size=N)
    - (5 * (1 - latent_risk_adj)),
    0, 180
).astype(int)

current_outstanding = np.clip(
    loan_amount * np.random.uniform(0.05, 0.9, N) * (0.6 + 0.7 * latent_risk_adj),
    0, loan_amount
).astype(int)

payment_frequency = np.random.choice(
    ["Monthly", "Bi-Weekly", "Weekly"], size=N, p=[0.75, 0.15, 0.10]
)

# Reliability & engagement scores (0-100), inversely related to latent risk
payment_reliability_score = np.clip(
    (95 - 70 * latent_risk_adj) + np.random.normal(0, 8, N), 5, 100
).astype(int)

engagement_score = np.clip(
    (85 - 55 * latent_risk_adj) + np.random.normal(0, 12, N), 5, 100
).astype(int)

# Did the customer respond positively to a previous reminder? (yes more likely if engaged)
previous_reminder_response = np.where(
    np.random.rand(N) < (engagement_score / 130), "Responded", "No Response"
)

# ----------------------------------------------------------------------
# 6. Upcoming obligation
# ----------------------------------------------------------------------
upcoming_installment_date_offset = np.random.randint(-15, 30, N)  # days from "today"
amount_due = np.clip(
    monthly_installment * np.random.uniform(0.9, 1.15, N), 500, 80000
).astype(int)

# ----------------------------------------------------------------------
# 7. Build the raw risk indicator (drives the label) from OBSERVABLE
#    columns (not directly from latent_risk) so the ML model has real
#    signal to learn from.
# ----------------------------------------------------------------------
def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

risk_indicator = (
    0.28 * minmax(days_past_due) +
    0.20 * minmax(previous_late_payments) +
    0.15 * minmax(previous_defaults) +
    0.12 * minmax(avg_payment_delay_days) +
    0.10 * minmax(current_outstanding) +
    0.10 * (1 - minmax(credit_score)) +
    0.05 * (1 - minmax(payment_reliability_score))
)

# add a little noise so boundaries aren't razor-sharp
risk_indicator = np.clip(risk_indicator + np.random.normal(0, 0.03, N), 0, 1)

# ----------------------------------------------------------------------
# 8. Convert risk_indicator into the 4 official labels using quantile
#    cuts, so we get a realistic class balance (most customers LOW/AT-RISK,
#    fewer CRITICAL) instead of an artificial 25/25/25/25 split.
# ----------------------------------------------------------------------
quantiles = risk_indicator.quantile([0.55, 0.80, 0.94]) if isinstance(risk_indicator, pd.Series) \
    else pd.Series(risk_indicator).quantile([0.55, 0.80, 0.94])

q55, q80, q94 = quantiles.iloc[0], quantiles.iloc[1], quantiles.iloc[2]

def label_from_indicator(x):
    if x <= q55:
        return "LOW"
    elif x <= q80:
        return "AT-RISK"
    elif x <= q94:
        return "HIGH"
    else:
        return "CRITICAL"

risk_category = np.array([label_from_indicator(x) for x in risk_indicator])

# ----------------------------------------------------------------------
# 9. Assemble final dataframe
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "customer_id": customer_id,
    "email": email,
    "phone": phone,
    "age": age,
    "employment_type": employment_type,
    "monthly_income": monthly_income,
    "account_age_months": account_age_months,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "monthly_installment": monthly_installment,
    "previous_late_payments": previous_late_payments,
    "avg_payment_delay_days": avg_payment_delay_days,
    "previous_defaults": previous_defaults,
    "days_past_due": days_past_due,
    "current_outstanding": current_outstanding,
    "payment_frequency": payment_frequency,
    "payment_reliability_score": payment_reliability_score,
    "engagement_score": engagement_score,
    "previous_reminder_response": previous_reminder_response,
    "amount_due": amount_due,
    "upcoming_installment_offset_days": upcoming_installment_date_offset,
    "risk_indicator_raw": risk_indicator.round(4) if isinstance(risk_indicator, pd.Series) else np.round(risk_indicator, 4),
    "risk_category": risk_category,
})

# A couple of realistic-but-harmless missing values (so preprocessing has something to do)
missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx, "engagement_score"] = np.nan

os.makedirs("data", exist_ok=True) if os.path.isdir("data") is False and os.path.basename(os.getcwd()) != "notebooks" else None

# Make sure we always write relative to the project root, regardless of
# whether this script is run from the project root or from notebooks/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

df.to_csv(os.path.join(data_dir, "customers.csv"), index=False)
print(f"Saved {len(df)} customers -> data/customers.csv")
print(df["risk_category"].value_counts())

# ----------------------------------------------------------------------
# 10. Simulated recovery / intervention outcomes
#     (used later by the Recovery Tracking page in Streamlit)
# ----------------------------------------------------------------------
np.random.seed(7)

intervention_map = {
    "LOW": "No Intervention",
    "AT-RISK": "Email Reminder",
    "HIGH": "Phone Call",
    "CRITICAL": "Manual Follow-up",
}

# Base recovery probability by intervention type (a phone call recovers
# more often than an email, manual follow-up highest but used only for
# the toughest cases so net rate is still lower)
base_recovery_prob = {
    "No Intervention": 0.92,
    "Email Reminder": 0.62,
    "Phone Call": 0.48,
    "Manual Follow-up": 0.35,
}

rows = []
for _, r in df.iterrows():
    intervention = intervention_map[r["risk_category"]]
    prob = base_recovery_prob[intervention]
    # more engaged customers recover more often
    prob = np.clip(prob + (r["engagement_score"] - 50) / 300 if not np.isnan(r["engagement_score"]) else prob, 0.02, 0.98)

    roll = np.random.rand()
    if roll < prob:
        status = "Recovered"
        recovered_amount = r["amount_due"]
    elif roll < prob + 0.20:
        status = "Partially Recovered"
        recovered_amount = int(r["amount_due"] * np.random.uniform(0.2, 0.7))
    elif roll < prob + 0.32:
        status = "Pending"
        recovered_amount = 0
    else:
        status = "Not Recovered"
        recovered_amount = 0

    rows.append({
        "customer_id": r["customer_id"],
        "risk_category": r["risk_category"],
        "intervention_type": intervention,
        "amount_due": r["amount_due"],
        "amount_recovered": recovered_amount,
        "outstanding_after": max(r["amount_due"] - recovered_amount, 0),
        "recovery_status": status,
        "communication_status": "Not Sent",
        "intervention_date_offset_days": int(np.random.randint(-10, 0)),
    })

recovery_df = pd.DataFrame(rows)
recovery_df.to_csv(os.path.join(data_dir, "recovery_data.csv"), index=False)
print(f"Saved recovery data -> data/recovery_data.csv")
print(recovery_df["recovery_status"].value_counts())
