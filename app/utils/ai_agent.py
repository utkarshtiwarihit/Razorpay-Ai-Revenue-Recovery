"""
AI Decision Engine
------------------
This is a deterministic, rule-based "AI Agent" (NOT an LLM call - no API
key or cost required). It looks at a customer's risk category, score and
context, and returns a recommended action, priority, reason and suggested
message. This keeps the demo 100% free and fully explainable.
"""

from .formatting import format_inr


def get_recommendation(customer: dict) -> dict:
    """
    customer: dict-like row with keys such as
        risk_category, risk_score, amount_due, days_past_due,
        previous_late_payments, engagement_score, current_outstanding,
        previous_reminder_response
    Returns a dict with: action, priority, reason, suggested_message,
    recovery_opportunity
    """
    risk_category = customer.get("risk_category_pred", customer.get("risk_category"))
    risk_score = customer.get("risk_score", 0)
    amount_due = customer.get("amount_due", 0)
    days_past_due = customer.get("days_past_due", 0)
    late_payments = customer.get("previous_late_payments", 0)
    engagement = customer.get("engagement_score", 50)
    outstanding = customer.get("current_outstanding", 0)
    responded_before = customer.get("previous_reminder_response", "No Response") == "Responded"

    reasons = []
    if days_past_due and days_past_due > 0:
        reasons.append(f"{int(days_past_due)} days past due")
    if late_payments and late_payments > 0:
        reasons.append(f"{int(late_payments)} previous late payment(s)")
    if outstanding and outstanding > 0:
        reasons.append(f"outstanding balance of {format_inr(outstanding)}")
    if engagement is not None and engagement < 40:
        reasons.append("low customer engagement history")
    if responded_before:
        reasons.append("has responded positively to reminders before")

    if risk_category == "CRITICAL":
        action = "Manual Manager Intervention"
        priority = "URGENT"
        reason = "Customer shows a high outstanding balance with repeated payment delays: " + \
                 ", ".join(reasons[:3]) + "." if reasons else \
                 "Customer profile indicates severe payment risk."
        suggested_message = (
            f"Please contact the customer directly regarding the overdue amount of "
            f"{format_inr(amount_due)} and discuss a suitable repayment plan."
        )
        recovery_opportunity = amount_due * 0.35  # manual follow-up recovers a smaller share

    elif risk_category == "HIGH":
        action = "Strong Reminder + Manager Review"
        priority = "HIGH PRIORITY"
        reason = "Customer has multiple recent delays and a growing outstanding balance: " + \
                 ", ".join(reasons[:3]) + "." if reasons else \
                 "Customer is trending toward higher risk."
        suggested_message = (
            f"Send a firm but polite reminder about the {format_inr(amount_due)} due, "
            f"and flag this account for a follow-up phone call."
        )
        recovery_opportunity = amount_due * 0.48

    elif risk_category == "AT-RISK":
        action = "Automated Payment Reminder"
        priority = "FOLLOW-UP"
        reason = "Early signs of risk detected: " + ", ".join(reasons[:3]) + "." if reasons else \
                 "Minor irregularities detected in recent payment behaviour."
        suggested_message = (
            f"Send a friendly automated reminder about the upcoming/overdue amount of "
            f"{format_inr(amount_due)}. Monitor the account closely over the next cycle."
        )
        recovery_opportunity = amount_due * 0.62

    else:  # LOW
        action = "No Action Required"
        priority = "MONITOR"
        reason = "Customer has a consistent payment history and strong reliability score."
        suggested_message = (
            "No reminder necessary. Optionally, send a friendly thank-you note "
            "to reinforce good payment behaviour."
        )
        recovery_opportunity = amount_due * 0.92

    return {
        "action": action,
        "priority": priority,
        "reason": reason,
        "suggested_message": suggested_message,
        "recovery_opportunity": round(recovery_opportunity, 2),
    }


def explain_risk_factors(customer: dict) -> list:
    """Return a small list of (factor, direction, value) tuples used to
    explain WHY a customer got their risk category - built from the same
    features the model was trained on (simple, honest explainability
    without needing SHAP)."""
    factors = []

    dpd = customer.get("days_past_due", 0)
    if dpd > 0:
        factors.append(("Days Past Due", "increases risk", f"{int(dpd)} days"))

    late = customer.get("previous_late_payments", 0)
    if late > 0:
        factors.append(("Previous Late Payments", "increases risk", f"{int(late)} times"))

    defaults = customer.get("previous_defaults", 0)
    if defaults > 0:
        factors.append(("Previous Defaults", "increases risk", f"{int(defaults)}"))

    outstanding = customer.get("current_outstanding", 0)
    if outstanding > 0:
        factors.append(("Outstanding Amount", "increases exposure", format_inr(outstanding)))

    credit = customer.get("credit_score", 700)
    if credit >= 750:
        factors.append(("Credit Score", "reduces risk", f"{int(credit)} (strong)"))
    elif credit < 600:
        factors.append(("Credit Score", "increases risk", f"{int(credit)} (weak)"))

    reliability = customer.get("payment_reliability_score", 70)
    if reliability >= 75:
        factors.append(("Payment Reliability", "reduces risk", f"{int(reliability)}/100 (strong)"))
    elif reliability < 40:
        factors.append(("Payment Reliability", "increases risk", f"{int(reliability)}/100 (weak)"))

    return factors
