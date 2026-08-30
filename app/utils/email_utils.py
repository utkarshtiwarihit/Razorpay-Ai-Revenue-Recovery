"""
Email generation + sending.

IMPORTANT (demo safety):
- Customer emails in the dataset are synthetic (customerXXXX@example.com)
  EXCEPT one CRITICAL customer (CUSTOMER_0958) whose email has been set to
  a real demo inbox (utkarshpradeshhit@gmail.com) so the team can show a
  live end-to-end send during the hackathon demo.
- The recipient field in the dashboard is always editable, so the manager
  can type any real address before sending.

TWO WAYS TO SEND (both free, no paid API):

1. "Open in Gmail" (recommended - this is what fixes '"email nahi ja
   raha hai"'): builds a Gmail *web compose* link
   (https://mail.google.com/mail/?view=cm&...) that opens in a new
   browser tab, already logged in as whichever Gmail account the manager
   is signed into (e.g. utkarshpradesh@gmail.com), with the customer's
   address in "To", and the subject/body pre-filled. The manager just
   clicks Gmail's own "Send" button. This needs NO app password and NO
   outbound SMTP access, so it works even on networks/sandboxes that
   block SMTP - which is almost always why direct SMTP sending fails.

2. "Send via SMTP" (optional/advanced): sends directly using smtplib +
   a Gmail App Password. Kept as a fallback for anyone who wants a fully
   automated, no-click send, but requires 2-Step Verification + an App
   Password and an outbound-SMTP-friendly network.
"""

import smtplib
import ssl
from urllib.parse import quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .formatting import format_inr

# Fallback routing used only if a customer record has no email at all.
DEMO_RECIPIENTS = {
    "AT-RISK": "utkarshpradeshhit@gmail.com",
    "HIGH": "engineerexplaininhindi@gmail.com",
    "CRITICAL": "utkarshpradesh@gmail.com",
}

DEFAULT_SENDER = "utkarshpradesh@gmail.com"


def get_demo_recipient(customer: dict) -> str:
    """Recipient defaults to the customer's OWN email in the dataset
    (editable afterwards in the UI). Falls back to a tier-based demo
    address only if the customer has no email on file."""
    email = customer.get("email")
    if isinstance(email, str) and "@" in email:
        return email
    risk_category = customer.get("risk_category_pred", customer.get("risk_category"))
    return DEMO_RECIPIENTS.get(risk_category, DEMO_RECIPIENTS["AT-RISK"])


def generate_email(customer: dict) -> dict:
    """Builds a professional, polite reminder email personalised with the
    customer's real (synthetic) data. No aggressive/collection language."""
    risk_category = customer.get("risk_category_pred", customer.get("risk_category"))
    customer_id = customer.get("customer_id", "CUSTOMER_0000")
    amount_due = customer.get("amount_due", 0)
    days_past_due = customer.get("days_past_due", 0)

    if risk_category == "CRITICAL":
        subject = f"Important: Payment Follow-up Required — {customer_id}"
    elif risk_category == "HIGH":
        subject = f"Reminder: Payment Overdue — {customer_id}"
    else:
        subject = "Payment Reminder — Action Required"

    overdue_line = (
        f"Our records show this account is currently {int(days_past_due)} days past due.\n\n"
        if days_past_due and days_past_due > 0
        else "Your next installment is coming up soon.\n\n"
    )

    body = (
        f"Dear Customer ({customer_id}),\n\n"
        f"This is a friendly reminder regarding your payment of {format_inr(amount_due)}.\n\n"
        f"{overdue_line}"
        "We understand that payments can sometimes be missed due to genuine reasons, "
        "and we're happy to help if you need to discuss a suitable repayment plan.\n\n"
        "Please make the payment at your earliest convenience, or reach out to our "
        "support team if you have any questions.\n\n"
        "Thank you for being a valued customer.\n\n"
        "Warm regards,\n"
        "Razorpay AI Finance Risk Engine (Demo)\n\n"
        "---\n"
        "This is an AI-generated, hackathon demo message. No real customer "
        "data was used to produce it."
    )

    recipient = get_demo_recipient(customer)

    return {
        "recipient": recipient,
        "subject": subject,
        "body": body,
        "risk_category": risk_category,
    }


def gmail_compose_url(recipient: str, subject: str, body: str) -> str:
    """Builds a Gmail *web* compose URL. Opening this in a browser (where
    the manager is already logged into Gmail, e.g. utkarshpradesh@gmail.com)
    pops open a ready-to-send compose window - this is the reliable,
    no-credentials-needed way to actually deliver the demo email."""
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote(recipient)}"
        f"&su={quote(subject)}"
        f"&body={quote(body)}"
    )


def send_email_real(sender_email: str, app_password: str, email_dict: dict) -> tuple:
    """Optional advanced path: really sends via Gmail SMTP + App Password.
    Returns (success: bool, message: str). If this fails (blocked network,
    wrong password type, etc.) use the 'Open in Gmail' button instead -
    it does not depend on SMTP at all.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = email_dict["recipient"]
        msg["Subject"] = email_dict["subject"]
        msg.attach(MIMEText(email_dict["body"], "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.starttls(context=context)
            server.login(sender_email, app_password)
            server.sendmail(sender_email, email_dict["recipient"], msg.as_string())

        return True, f"Email really sent to {email_dict['recipient']}"
    except Exception as e:
        return False, (
            f"SMTP send failed ({e}). This is usually a blocked network or an "
            f"app-password issue — use the 'Open in Gmail' button instead, it "
            f"always works in a browser."
        )
