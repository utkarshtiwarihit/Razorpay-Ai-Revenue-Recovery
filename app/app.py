"""
Razorpay AI Finance Risk Engine - Streamlit Dashboard
------------------------------------------------------
Run from the project root with:
    streamlit run app/app.py

This file ONLY loads pre-computed data + a pre-trained model
(see notebooks/generate_data.py and notebooks/train_model.py).
It does NOT train anything itself.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.formatting import format_inr, format_inr_compact, inr_grouping
from utils.ai_agent import get_recommendation, explain_risk_factors
from utils.email_utils import generate_email, send_email_real, gmail_compose_url, DEMO_RECIPIENTS, DEFAULT_SENDER

# ----------------------------------------------------------------------
# PAGE CONFIG + GLOBAL STYLE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Razorpay AI Finance Risk Engine",
    page_icon="\U0001F4B3",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "LOW": "#22c55e",
    "AT-RISK": "#f59e0b",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #0b1220 100%);
}

/* Hide default streamlit chrome for a cleaner "product" feel */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #0891b2 100%);
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(79,70,229,0.35);
}
.hero h1 {
    color: white; font-size: 30px; font-weight: 800; margin: 0;
}
.hero p {
    color: #e0e7ff; font-size: 15px; margin: 4px 0 0 0;
}

.kpi-card {
    border-radius: 16px;
    padding: 18px 20px;
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    height: 108px;
}
.kpi-label { font-size: 12.5px; opacity: 0.9; font-weight: 500; letter-spacing: 0.3px;}
.kpi-value { font-size: 25px; font-weight: 800; margin-top: 6px; }
.kpi-sub { font-size: 11px; opacity: 0.85; margin-top: 4px;}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

.badge {
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-weight: 700; font-size: 12.5px; color: white;
}

.action-item {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #7c3aed;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.stButton>button {
    border-radius: 10px; font-weight: 600; border: none;
    background: linear-gradient(120deg, #4f46e5, #7c3aed);
    color: white;
}
.stButton>button:hover { opacity: 0.9; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@st.cache_data
def load_data():
    scored_path = os.path.join(DATA_DIR, "customers_scored.csv")
    raw_path = os.path.join(DATA_DIR, "customers.csv")
    recovery_path = os.path.join(DATA_DIR, "recovery_data.csv")

    if os.path.exists(scored_path):
        customers = pd.read_csv(scored_path)
    elif os.path.exists(raw_path):
        customers = pd.read_csv(raw_path)
        customers["risk_category_pred"] = customers["risk_category"]
        customers["risk_score"] = customers["risk_indicator_raw"] * 100
    else:
        return None, None

    if os.path.exists(recovery_path):
        recovery = pd.read_csv(recovery_path)
    else:
        recovery = pd.DataFrame(columns=[
            "customer_id", "risk_category", "intervention_type", "amount_due",
            "amount_recovered", "outstanding_after", "recovery_status",
            "communication_status"
        ])

    return customers, recovery


customers_df, recovery_df = load_data()

if customers_df is None:
    st.error(
        "No dataset found. Please run `python notebooks/generate_data.py` "
        "and `python notebooks/train_model.py` from the project root first."
    )
    st.stop()

# Keep an editable, session-persisted copy of recovery data so buttons
# (Simulate Send / Mark Contacted etc.) can update status live in the demo.
if "recovery_state" not in st.session_state:
    st.session_state.recovery_state = recovery_df.copy()

recovery_state = st.session_state.recovery_state

# risk_category used everywhere below = the MODEL's predicted category
if "risk_category_pred" not in customers_df.columns:
    customers_df["risk_category_pred"] = customers_df["risk_category"]


# ----------------------------------------------------------------------
# LOGIN
# ----------------------------------------------------------------------
def login_screen():
    st.markdown(
        """
        <div class="hero" style="text-align:center; margin-top: 60px;">
            <h1>\U0001F4B3 Razorpay AI Finance Risk Engine</h1>
            <p>AI-powered payment risk & recovery intelligence — Manager Login</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="manager")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            if st.button("Login", use_container_width=True):
                if username.strip() == "manager" and password == "razorpay123":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try username: manager / password: razorpay123")
            st.caption("Demo credentials — username: `manager`, password: `razorpay123`. "
                       "This is demo-only authentication, not production security.")
            st.markdown('</div>', unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
    st.stop()


# ----------------------------------------------------------------------
# SHARED HELPERS
# ----------------------------------------------------------------------
def risk_badge(category: str) -> str:
    color = RISK_COLORS.get(category, "#6b7280")
    return f'<span class="badge" style="background:{color}">{category}</span>'


def kpi_card(label, value, sub, gradient):
    st.markdown(
        f"""
        <div class="kpi-card" style="background:{gradient};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def merged_data():
    """customers joined with their (mutable) recovery/communication state."""
    return customers_df.merge(
        st.session_state.recovery_state[
            ["customer_id", "intervention_type", "amount_recovered",
             "outstanding_after", "recovery_status", "communication_status"]
        ],
        on="customer_id", how="left"
    )


# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION  (Analytics page removed per requirement)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001F4B3 Razorpay Risk Engine")
    st.caption("AI-powered risk & recovery")
    page = st.radio(
        "Navigate",
        ["\U0001F3E0 Dashboard", "\U0001F465 Customer Risk", "\U0001F916 AI Actions",
         "\U0001F4B0 Recovery", "\U0001F4E7 Communication"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Logged in as **manager**")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

st.markdown(
    """
    <div class="hero">
        <h1>Razorpay AI Finance Risk Engine</h1>
        <p>AI-powered payment risk & recovery intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

data = merged_data()


# ----------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------
if page.endswith("Dashboard"):
    counts = data["risk_category_pred"].value_counts()
    total_customers = len(data)
    total_due = data["amount_due"].sum()
    amount_at_risk = data.loc[data["risk_category_pred"].isin(["AT-RISK", "HIGH", "CRITICAL"]), "amount_due"].sum()
    recovered = data["amount_recovered"].fillna(0).sum()
    recovery_rate = (recovered / total_due * 100) if total_due > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("TOTAL CUSTOMERS", f"{total_customers:,}", "in portfolio",
                  "linear-gradient(120deg,#4f46e5,#6366f1)")
    with c2:
        kpi_card("TOTAL AMOUNT DUE", format_inr_compact(total_due), "across all customers",
                  "linear-gradient(120deg,#0891b2,#0ea5e9)")
    with c3:
        kpi_card("AMOUNT AT RISK", format_inr_compact(amount_at_risk), "AT-RISK + HIGH + CRITICAL",
                  "linear-gradient(120deg,#f59e0b,#f97316)")
    with c4:
        kpi_card("RECOVERY RATE", f"{recovery_rate:.1f}%", f"{format_inr_compact(recovered)} recovered",
                  "linear-gradient(120deg,#16a34a,#22c55e)")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, cat, grad in zip(
        [c1, c2, c3, c4],
        ["LOW", "AT-RISK", "HIGH", "CRITICAL"],
        ["linear-gradient(120deg,#16a34a,#22c55e)", "linear-gradient(120deg,#d97706,#f59e0b)",
         "linear-gradient(120deg,#ea580c,#f97316)", "linear-gradient(120deg,#dc2626,#ef4444)"]
    ):
        with col:
            n = int(counts.get(cat, 0))
            pct = (n / total_customers * 100) if total_customers else 0
            kpi_card(cat, f"{n:,}", f"{pct:.1f}% of customers", grad)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Risk Distribution")
        fig = px.pie(
            names=counts.index, values=counts.values, hole=0.55,
            color=counts.index, color_discrete_map=RISK_COLORS,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(
            showlegend=True, height=340, paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Amount Due by Risk Category")
        amt_by_cat = data.groupby("risk_category_pred")["amount_due"].sum().reindex(
            ["LOW", "AT-RISK", "HIGH", "CRITICAL"]
        )
        fig2 = go.Figure(go.Bar(
            x=amt_by_cat.index, y=amt_by_cat.values,
            marker_color=[RISK_COLORS[c] for c in amt_by_cat.index],
            text=[format_inr_compact(v) for v in amt_by_cat.values],
            textposition="outside",
        ))
        fig2.update_layout(
            height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(t=10, b=10, l=10, r=10),
            yaxis_title=None, xaxis_title=None,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Risk Score Distribution")
    fig3 = px.histogram(
        data, x="risk_score", nbins=30, color="risk_category_pred",
        color_discrete_map=RISK_COLORS,
    )
    fig3.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", margin=dict(t=10, b=10, l=10, r=10),
        legend_title_text="", xaxis_title="Risk Score (0-100)", yaxis_title="Customers",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE: CUSTOMER RISK
# ----------------------------------------------------------------------
elif page.endswith("Customer Risk"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("\U0001F50D Search Customer ID", placeholder="e.g. CUSTOMER_0732")
    with col2:
        risk_filter = st.selectbox("Risk Category", ["All", "LOW", "AT-RISK", "HIGH", "CRITICAL"])
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = data.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["risk_category_pred"] == risk_filter]
    if search:
        filtered = filtered[filtered["customer_id"].str.contains(search.strip(), case=False, na=False)]

    st.caption(f"Showing **{len(filtered):,}** of {len(data):,} customers")

    display_cols = [
        "customer_id", "risk_category_pred", "risk_score", "amount_due",
        "days_past_due", "credit_score", "previous_late_payments",
        "payment_reliability_score", "recovery_status", "communication_status",
    ]
    pretty = filtered[display_cols].rename(columns={
        "customer_id": "Customer ID", "risk_category_pred": "Risk Category",
        "risk_score": "Risk Score", "amount_due": "Amount Due",
        "days_past_due": "Days Past Due", "credit_score": "Credit Score",
        "previous_late_payments": "Late Payments", "payment_reliability_score": "Reliability",
        "recovery_status": "Recovery Status", "communication_status": "Communication Status",
    }).sort_values("Risk Score", ascending=False)
    pretty["Amount Due"] = pretty["Amount Due"].apply(format_inr)

    st.dataframe(pretty, use_container_width=True, height=420, hide_index=True)

    st.markdown("### \U0001F50E Customer Detail")
    chosen = st.selectbox("Select a customer to view full profile", filtered["customer_id"].tolist())

    if chosen:
        row = data[data["customer_id"] == chosen].iloc[0].to_dict()
        cat = row["risk_category_pred"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        top1, top2, top3, top4 = st.columns(4)
        with top1:
            st.markdown(f"**{chosen}**", unsafe_allow_html=True)
            st.markdown(risk_badge(cat), unsafe_allow_html=True)
        with top2:
            st.metric("Risk Score", f"{row['risk_score']:.0f}/100")
        with top3:
            st.metric("Amount Due", format_inr(row["amount_due"]))
        with top4:
            st.metric("Days Past Due", int(row["days_past_due"]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Credit Score", int(row["credit_score"]))
        c2.metric("Previous Late Payments", int(row["previous_late_payments"]))
        c3.metric("Payment Reliability", f"{int(row['payment_reliability_score'])}/100")
        c4.metric("Outstanding", format_inr(row["current_outstanding"]))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Why is this customer classified as " + cat + "?")
        factors = explain_risk_factors(row)
        if factors:
            for name, direction, value in factors:
                arrow = "\U0001F53A" if "increase" in direction else "\U0001F53B"
                st.markdown(f"{arrow} **{name}** — {value} ({direction})")
        else:
            st.write("No significant risk factors detected.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### \U0001F916 AI Recommendation")
        rec = get_recommendation(row)
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(f"**Priority:** {rec['priority']}")
            st.markdown(f"**Recommended Action:** {rec['action']}")
            st.markdown(f"**Expected Recovery Opportunity:** {format_inr(rec['recovery_opportunity'])}")
        with rc2:
            st.markdown(f"**Reason:** {rec['reason']}")
            st.markdown(f"**Suggested Next Step:** {rec['suggested_message']}")
        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE: AI ACTIONS
# ----------------------------------------------------------------------
elif page.endswith("AI Actions"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### \U0001F916 AI Action Queue")

    urgent = data[data["risk_category_pred"] == "CRITICAL"]
    high_priority = data[data["risk_category_pred"] == "HIGH"]
    follow_up = data[data["risk_category_pred"] == "AT-RISK"]
    monitor = data[data["risk_category_pred"] == "LOW"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001F534 URGENT", len(urgent), "Manual intervention")
    c2.metric("\U0001F7E0 HIGH PRIORITY", len(high_priority), "Manager review")
    c3.metric("\U0001F7E1 FOLLOW-UP", len(follow_up), "Automated reminder")
    c4.metric("\U0001F7E2 MONITOR", len(monitor), "No action needed")

    recoverable = data.loc[data["risk_category_pred"].isin(["HIGH", "CRITICAL"]), "amount_due"].sum() * 0.45
    st.info(f"\U0001F4B0 **{format_inr_compact(recoverable)}** potentially recoverable "
            f"from HIGH & CRITICAL accounts with timely intervention.")
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["\U0001F534 URGENT", "\U0001F7E0 HIGH PRIORITY", "\U0001F7E1 FOLLOW-UP"])
    for tab, subset, label in zip(tabs, [urgent, high_priority, follow_up],
                                   ["CRITICAL", "HIGH", "AT-RISK"]):
        with tab:
            subset_sorted = subset.sort_values("risk_score", ascending=False).head(25)
            for _, r in subset_sorted.iterrows():
                rec = get_recommendation(r.to_dict())
                st.markdown(
                    f"""
                    <div class="action-item">
                        <b>{r['customer_id']}</b> {risk_badge(label)}
                        &nbsp;|&nbsp; Score: {r['risk_score']:.0f}
                        &nbsp;|&nbsp; Due: {format_inr(r['amount_due'])}
                        &nbsp;|&nbsp; {int(r['days_past_due'])} days overdue<br>
                        <span style="opacity:0.85;">→ {rec['action']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------
# PAGE: RECOVERY
# ----------------------------------------------------------------------
elif page.endswith("Recovery"):
    total_due = data["amount_due"].sum()
    recovered = data["amount_recovered"].fillna(0).sum()
    outstanding = data["outstanding_after"].fillna(data["amount_due"]).sum()
    rate = (recovered / total_due * 100) if total_due > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    kpi_card_kwargs = [
        ("TOTAL AMOUNT DUE", format_inr_compact(total_due), "portfolio-wide", "linear-gradient(120deg,#4f46e5,#6366f1)"),
        ("AMOUNT RECOVERED", format_inr_compact(recovered), "via all interventions", "linear-gradient(120deg,#16a34a,#22c55e)"),
        ("OUTSTANDING", format_inr_compact(outstanding), "still pending", "linear-gradient(120deg,#f59e0b,#f97316)"),
        ("RECOVERY RATE", f"{rate:.1f}%", "recovered / due", "linear-gradient(120deg,#0891b2,#0ea5e9)"),
    ]
    for col, args in zip([c1, c2, c3, c4], kpi_card_kwargs):
        with col:
            kpi_card(*args)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Recovery by Risk Category")
        rec_by_cat = data.groupby("risk_category_pred").apply(
            lambda g: pd.Series({
                "Due": g["amount_due"].sum(),
                "Recovered": g["amount_recovered"].fillna(0).sum(),
            })
        ).reindex(["LOW", "AT-RISK", "HIGH", "CRITICAL"])
        fig = go.Figure()
        fig.add_bar(name="Amount Due", x=rec_by_cat.index, y=rec_by_cat["Due"], marker_color="#4f46e5")
        fig.add_bar(name="Amount Recovered", x=rec_by_cat.index, y=rec_by_cat["Recovered"], marker_color="#22c55e")
        fig.update_layout(
            barmode="group", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Recovery by Intervention Type (Leaderboard)")
        by_intervention = data.groupby("intervention_type").apply(
            lambda g: pd.Series({
                "recovery_rate": (g["amount_recovered"].fillna(0).sum() / g["amount_due"].sum() * 100)
                if g["amount_due"].sum() > 0 else 0
            })
        ).sort_values("recovery_rate", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=by_intervention["recovery_rate"], y=by_intervention.index, orientation="h",
            marker_color="#7c3aed",
            text=[f"{v:.1f}%" for v in by_intervention["recovery_rate"]], textposition="outside",
        ))
        fig2.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(t=10, b=10), xaxis_title="Recovery Rate (%)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Before vs After Intervention")
    exposure_before = data.loc[data["risk_category_pred"].isin(["AT-RISK", "HIGH", "CRITICAL"]), "amount_due"].sum()
    exposure_after = data.loc[data["risk_category_pred"].isin(["AT-RISK", "HIGH", "CRITICAL"]), "outstanding_after"].fillna(0).sum()
    resolved_customers = (data["recovery_status"] == "Recovered").sum()

    b1, b2, b3 = st.columns(3)
    b1.metric("Risk Exposure Before", format_inr_compact(exposure_before))
    b2.metric("Risk Exposure After", format_inr_compact(exposure_after),
              delta=f"-{format_inr_compact(exposure_before - exposure_after)}", delta_color="inverse")
    b3.metric("Customers Resolved", f"{resolved_customers:,}")
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE: COMMUNICATION
# ----------------------------------------------------------------------
elif page.endswith("Communication"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### \U0001F4E7 AI Email Generator")
    st.caption(
        "Recipient defaults to the customer's email on file — edit it to any real "
        "address before sending. One CRITICAL customer (**CUSTOMER_0958**) is pre-set "
        f"to a real demo inbox (`{DEMO_RECIPIENTS['AT-RISK']}`) for a live demo."
    )

    eligible = data[data["risk_category_pred"].isin(["AT-RISK", "HIGH", "CRITICAL"])]
    chosen = st.selectbox("Select a customer", eligible["customer_id"].tolist())
    st.markdown('</div>', unsafe_allow_html=True)

    if chosen:
        row = data[data["customer_id"] == chosen].iloc[0].to_dict()
        cat = row["risk_category_pred"]
        email = generate_email(row)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        left, right = st.columns([1, 2])
        with left:
            st.markdown(risk_badge(cat), unsafe_allow_html=True)
            st.markdown(f"**Amount Due:** {format_inr(row['amount_due'])}")
            st.markdown(f"**Days Past Due:** {int(row['days_past_due'])}")
            if cat == "CRITICAL":
                st.warning("\u26A0\uFE0F CRITICAL customers are normally escalated for **manual "
                           "manager action** — email is still available below for demo purposes.")
        with right:
            recipient = st.text_input(
                "Recipient (edit to any real email before sending)",
                value=email["recipient"], key=f"recipient_{chosen}",
            )
            subject = st.text_input("Subject", value=email["subject"], key=f"subject_{chosen}")
            body = st.text_area("Body", value=email["body"], height=220, key=f"body_{chosen}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### \u2709\uFE0F Send this email")
        st.caption(
            "**Recommended:** click **Open in Gmail** — it opens Gmail's compose window in a "
            "new tab, already logged in as your own Gmail account, with the customer's address, "
            "subject and message pre-filled. Just hit Gmail's own Send button. This always works "
            "and needs no password/app-password setup, unlike direct SMTP sending."
        )
        sender_email = st.text_input(
            "Your Gmail address (the account you're logged into in your browser)",
            value=DEFAULT_SENDER, key="sender_email",
        )

        compose_url = gmail_compose_url(recipient, subject, body)

        def update_status(customer_id, status):
            st.session_state.recovery_state.loc[
                st.session_state.recovery_state["customer_id"] == customer_id,
                "communication_status"
            ] = status

        btn1, btn2 = st.columns(2)
        with btn1:
            st.link_button("\U0001F4E4 Open in Gmail (Compose & Send)", compose_url,
                            use_container_width=True)
            if st.button("I sent it — mark as Sent", use_container_width=True, key=f"marksent_{chosen}"):
                update_status(chosen, "Sent (via Gmail)")
                st.success(f"{chosen} marked as sent to {recipient}.")
        with btn2:
            simulate_clicked = st.button("Simulate Send (no real email)", use_container_width=True,
                                          key=f"sim_{chosen}")
            contacted_clicked = st.button("Mark as Contacted (Manual)", use_container_width=True,
                                           key=f"contact_{chosen}")

        st.markdown("###### Advanced: send automatically via SMTP (optional)")
        st.caption(
            "Needs a Gmail **App Password** (Google Account → Security → 2-Step Verification → "
            "App Passwords) and a network that allows outbound SMTP. If this fails, use "
            "'Open in Gmail' above instead — that method doesn't depend on SMTP at all."
        )
        app_password = st.text_input("Gmail App Password", type="password", key="app_password")
        real_clicked = st.button("Send via SMTP", key=f"smtp_{chosen}")

        if simulate_clicked:
            update_status(chosen, "Sent (Simulated)")
            st.success(f"Simulated: reminder marked as sent for {chosen}.")

        if real_clicked:
            if not sender_email or not app_password:
                st.error("Enter your Gmail address and App Password, or use 'Open in Gmail' instead.")
            else:
                success, message = send_email_real(
                    sender_email, app_password,
                    {"recipient": recipient, "subject": subject, "body": body}
                )
                if success:
                    update_status(chosen, "Sent (Real Email)")
                    st.success(message)
                else:
                    st.warning(message)

        if contacted_clicked:
            update_status(chosen, "Manually Contacted")
            st.success(f"{chosen} marked as manually contacted.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Communication Status Overview")
    status_counts = st.session_state.recovery_state["communication_status"].value_counts()
    fig = px.bar(
        x=status_counts.index, y=status_counts.values,
        color=status_counts.index,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", showlegend=False, xaxis_title=None, yaxis_title="Customers",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
