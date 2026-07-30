import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import os

import joblib
import pandas as pd
import psycopg2
import psycopg2.extras
import shap
import streamlit as st
from dotenv import load_dotenv
from features import get_connection
from train import FEATURE_COLUMNS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")

st.set_page_config(page_title="Fraud Decision Dashboard", layout="wide")


def check_password():
    def password_entered():
        if st.session_state["password_input"] == os.getenv("DASHBOARD_PASSWORD"):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #f4f6f9 0%, #e9edf2 100%);
            }
            .brand-banner {
                background: linear-gradient(135deg, #001f3f 0%, #003366 100%);
                padding: 28px 20px;
                border-radius: 12px 12px 0 0;
                text-align: center;
            }
            .brand-mark {
                display: inline-block;
                width: 44px;
                height: 44px;
                border: 3px solid white;
                border-radius: 10px;
                color: white;
                font-size: 18px;
                font-weight: 800;
                line-height: 38px;
                margin-bottom: 10px;
            }
            .brand-name {
                color: white;
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            .brand-tagline {
                color: #a9c1dd;
                font-size: 12px;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-top: 2px;
            }
            .login-card {
                background: white;
                padding: 32px 40px 32px 40px;
                border-radius: 0 0 12px 12px;
                box-shadow: 0 4px 24px rgba(0, 31, 63, 0.12);
                text-align: center;
            }
            .stButton button {
                background-color: #001f3f;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 0;
                width: 100%;
                font-weight: 600;
            }
            .stButton button:hover {
                background-color: #003366;
                color: white;
                border: none;
            }
            div[data-testid="stTextInput"] input {
                border-radius: 6px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="brand-banner">
                    <div class="brand-mark">SFI</div>
                    <div class="brand-name">SENTINEL FRAUD INTELLIGENCE</div>
                    <div class="brand-tagline">Real-Time Risk & Case Management</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown("**Analyst Access Portal**")
            st.text_input("Password", type="password", key="password_input", label_visibility="collapsed", placeholder="Enter your password")
            st.button("Sign In", on_click=password_entered)
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("Incorrect password. Please try again.")
            st.markdown('<div style="font-size:12px;color:#9ca3af;margin-top:20px;">Authorized personnel only</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        return False
    else:
        return True


DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

if not DEMO_MODE:
    if not check_password():
        st.stop()


@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    explainer = shap.TreeExplainer(model)
    return model, explainer


def get_recent_activity(limit=15):
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT transactionid, transactionamt, fraud_probability, was_flagged, processed_at "
        f"FROM activity_log ORDER BY processed_at DESC LIMIT {limit};",
        conn,
    )
    conn.close()
    return df


def get_pending_count():
    conn = get_connection()
    result = pd.read_sql(
        """
        SELECT COUNT(*) AS pending_count
        FROM flagged_cases f
        LEFT JOIN analyst_decisions d ON f.transactionid = d.transactionid
        WHERE d.id IS NULL;
        """,
        conn,
    )
    conn.close()
    return int(result["pending_count"].iloc[0])


def get_pending_value():
    conn = get_connection()
    result = pd.read_sql(
        """
        SELECT COALESCE(SUM(t.transactionamt), 0) AS total_pending_value
        FROM flagged_cases f
        JOIN transactions t ON f.transactionid = t.transactionid
        LEFT JOIN analyst_decisions d ON f.transactionid = d.transactionid
        WHERE d.id IS NULL;
        """,
        conn,
    )
    conn.close()
    return result["total_pending_value"].iloc[0]


def get_queue():
    conn = get_connection()
    query = """
        SELECT
            f.transactionid,
            f.fraud_probability,
            f.requires_second_approval,
            t.transactionamt,
            t.transactiondt,
            t.productcd,
            t.card1,
            t.card4,
            t.card6,
            t.addr1,
            t.addr2,
            t.dist1,
            t.p_emaildomain,
            t.r_emaildomain,
            i.devicetype,
            i.deviceinfo
        FROM flagged_cases f
        JOIN transactions t ON f.transactionid = t.transactionid
        LEFT JOIN identities i ON f.transactionid = i.transactionid
        LEFT JOIN analyst_decisions d ON f.transactionid = d.transactionid
        WHERE d.id IS NULL
        ORDER BY f.fraud_probability DESC
        LIMIT 100;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_case_features(transaction_id):
    conn = get_connection()
    query = f"""
        SELECT card1, transactionamt, has_identity,
               uid_avg_amt_before_this, uid_txn_count_before_this,
               seconds_since_last_txn, email_txn_count_before_this,
               account_age_days, identity_linkage_score, match_flag_count,
               hour_of_day, day_of_week, no_identity_high_velocity
        FROM (
            SELECT
                t.transactionid, t.card1, t.transactionamt,
                CASE WHEN i.transactionid IS NOT NULL THEN 1 ELSE 0 END AS has_identity,
                CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA')) AS uid,
                AVG(t.transactionamt) OVER (
                    PARTITION BY CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA'))
                    ORDER BY t.transactiondt
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS uid_avg_amt_before_this,
                COUNT(*) OVER (
                    PARTITION BY CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA'))
                    ORDER BY t.transactiondt
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS uid_txn_count_before_this,
                t.transactiondt - LAG(t.transactiondt) OVER (
                    PARTITION BY CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA'))
                    ORDER BY t.transactiondt
                ) AS seconds_since_last_txn,
                COUNT(*) OVER (
                    PARTITION BY t.p_emaildomain ORDER BY t.transactiondt
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS email_txn_count_before_this,
                COALESCE(t.d1, 0) AS account_age_days,
                COALESCE(t.c1, 0) + COALESCE(t.c2, 0) AS identity_linkage_score,
                (CASE WHEN t.m1 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) +
                (CASE WHEN t.m2 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) +
                (CASE WHEN t.m3 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) +
                (CASE WHEN t.m4 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) AS match_flag_count,
                MOD(t.transactiondt / 3600, 24) AS hour_of_day,
                MOD(t.transactiondt / 86400, 7) AS day_of_week,
                CASE
                    WHEN (CASE WHEN i.transactionid IS NOT NULL THEN 1 ELSE 0 END) = 0
                         AND (t.transactiondt - LAG(t.transactiondt) OVER (
                                PARTITION BY CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA'))
                                ORDER BY t.transactiondt
                              )) < 300
                    THEN 1 ELSE 0
                END AS no_identity_high_velocity
            FROM transactions t
            LEFT JOIN identities i ON t.transactionid = i.transactionid
        ) sub
        WHERE transactionid = {transaction_id};
    """
    df = pd.read_sql(query, conn)
    conn.close()

    numeric_columns = [
        "uid_avg_amt_before_this",
        "uid_txn_count_before_this",
        "seconds_since_last_txn",
        "email_txn_count_before_this",
        "account_age_days",
        "identity_linkage_score",
        "match_flag_count",
        "hour_of_day",
        "day_of_week",
        "no_identity_high_velocity",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[FEATURE_COLUMNS]


def save_decision(transaction_id, analyst_name, decision):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO analyst_decisions (transactionid, analyst_name, decision) VALUES (%s, %s, %s)",
            (transaction_id, analyst_name, decision),
        )
    conn.commit()
    conn.close()

FEATURE_DISPLAY_NAMES = {
    "transactionamt": "Transaction Amount",
    "has_identity": "Identity Verification on File",
    "uid_avg_amt_before_this": "Typical Spend for This Customer",
    "uid_txn_count_before_this": "Prior Transaction Count",
    "seconds_since_last_txn": "Time Since Last Purchase",
    "email_txn_count_before_this": "Email Domain Activity Level",
    "account_age_days": "Account Age (Days)",
    "identity_linkage_score": "Linked Addresses/Devices Count",
    "match_flag_count": "Failed Identity Checks",
    "hour_of_day": "Time-of-Day Pattern",
    "day_of_week": "Day-of-Week Pattern",
    "no_identity_high_velocity": "Rapid Transaction with No ID",
    "card1": "Card Reference Number",
}




def score_a_new_transaction(card1, addr1, transactionamt, transactiondt, p_emaildomain):
    """
    Takes the raw facts of a brand new transaction, computes its real
    features from database history, and scores it - this is the same
    logic app.py exposes as an API, reused directly here so the
    dashboard can demonstrate live scoring without a separate service.
    """
    import psycopg2.extras

    conn = get_connection()

    with conn.cursor() as cur:
        next_id_query = "SELECT COALESCE(MAX(transactionid), 3000000) + 1 FROM transactions;"
        cur.execute(next_id_query)
        new_transaction_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO transactions
                (transactionid, isfraud, transactiondt, transactionamt,
                 productcd, card1, addr1, p_emaildomain)
            VALUES (%s, 0, %s, %s, 'W', %s, %s, %s)
            ON CONFLICT (transactionid) DO NOTHING
            """,
            (new_transaction_id, transactiondt, transactionamt, card1, addr1, p_emaildomain),
        )
    conn.commit()

    feature_row = get_case_features(new_transaction_id)
    conn.close()

    probability = model.predict_proba(feature_row)[:, 1][0]
    is_flagged = bool(probability >= 0.04)

    shap_values = explainer.shap_values(feature_row)
    contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

    return {
        "transaction_id": new_transaction_id,
        "fraud_probability": probability,
        "flagged": is_flagged,
        "contributions": contributions,
        "feature_row": feature_row,
    }                


def format_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return value


def kpi_card(label, value, alert=False):
    css_class = "kpi-card alert" if alert else "kpi-card"
    st.markdown(
        f'<div class="{css_class}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 18px 16px;
        box-shadow: 0 2px 10px rgba(0,31,63,0.08);
        border-left: 4px solid #001f3f;
        margin-bottom: 8px;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label {
        color: #6b7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #001f3f;
        font-size: 22px;
        font-weight: 700;
        white-space: nowrap;
    }
    .kpi-card.alert { border-left-color: #b91c1c; }
    .kpi-card.alert .kpi-value { color: #b91c1c; }
    .sidebar-header {
        color: #001f3f;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;    
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    with st.container(border=True):
        st.markdown('<div class="sidebar-header">Analyst</div>', unsafe_allow_html=True)
        analyst_name = st.text_input("Analyst name", value="Shubh Keshri", label_visibility="collapsed")

    if st.button("Logout"):
        st.session_state["password_correct"] = False
        st.rerun()


main_col, activity_col = st.columns([3, 1])

with main_col:
    st.title("Fraud Analyst Dashboard")

    model, explainer = load_model()
    queue = get_queue()

    with st.container(border=True):
        st.markdown('<div class="sidebar-header">Case Lookup & Filters</div>', unsafe_allow_html=True)
        search_col, risk_col, amount_col = st.columns(3)
        with search_col:
            search_id = st.text_input("Search by Transaction ID", placeholder="Enter transaction ID")
        with risk_col:
            min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)
        with amount_col:
            min_amount = st.number_input("Minimum amount (GBP)", min_value=0.0, value=0.0, step=50.0)

    if search_id:
        try:
            search_id_int = int(search_id)
            queue = queue[queue["transactionid"] == search_id_int]
            if len(queue) == 0:
                st.warning("No matching case found in the current pending queue.")
        except ValueError:
            st.warning("Please enter a valid numeric Transaction ID.")

    queue = queue[queue["fraud_probability"] >= min_risk]
    queue = queue[queue["transactionamt"] >= min_amount]

    total_value_at_risk = queue["transactionamt"].sum()
    avg_risk = queue["fraud_probability"].mean() if len(queue) > 0 else 0
    pending_count = get_pending_count()
    pending_value = get_pending_value()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        kpi_card("Cases in Queue", f"{len(queue):,}")
    with kpi2:
        kpi_card("Value at Risk", f"GBP {total_value_at_risk/1000:,.1f}K")
    with kpi3:
        kpi_card("Avg Risk Score", f"{avg_risk:.1%}", alert=(avg_risk > 0.5))
    with kpi4:
        kpi_card("Pending Review", f"{pending_count:,}", alert=(pending_count > 1000))
    with kpi5:
        kpi_card("Pending Value", f"GBP {pending_value/1000:,.1f}K")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Live Transaction Scoring - Test a New Transaction", expanded=False):
        st.markdown("Enter transaction details below to see the model score it in real time.")

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            live_card1 = st.number_input("Card Number", min_value=1000, max_value=20000, value=5000)
            live_amount = st.number_input("Transaction Amount (GBP)", min_value=0.0, value=100.0)
        with lc2:
            live_addr1 = st.number_input("Billing Address Code", min_value=100, max_value=540, value=200)
            live_time = st.number_input("Time Offset (seconds)", min_value=0, value=16000000)
        with lc3:
            live_email = st.selectbox("Email Domain", ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"])

        if st.button("Score This Transaction", key="live_score_button"):
            with st.spinner("Analyzing transaction..."):
                result = score_a_new_transaction(
                    live_card1, live_addr1, live_amount, live_time, live_email
                )

            st.markdown("---")
            risk_col1, risk_col2 = st.columns(2)
            with risk_col1:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
            with risk_col2:
                verdict = "FLAGGED FOR REVIEW" if result["flagged"] else "APPEARS LEGITIMATE"
                st.metric("Verdict", verdict)

            st.markdown("**Why:**")
            for feature_name, contribution in result["contributions"][:5]:
                display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
                direction = "increases risk" if contribution > 0 else "decreases risk"
                st.write(f"- {display_name}: {direction} ({contribution:+.3f})")

    for _, case in queue.iterrows():
        approval_tag = " [SECOND APPROVAL REQUIRED]" if case["requires_second_approval"] else ""
        header = f"Txn {case['transactionid']} - {case['fraud_probability']:.1%} risk - GBP {case['transactionamt']:.2f}{approval_tag}"

        with st.expander(header):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Transaction**")
                st.write(f"Amount: GBP {case['transactionamt']:.2f}")
                st.write(f"Product category: {format_value(case['productcd'])}")
                total_seconds = int(case['transactiondt'])
                days = total_seconds // 86400
                hours = (total_seconds % 86400) // 3600
                minutes = (total_seconds % 3600) // 60
                st.write(f"Transaction Time: Day {days}, {hours:02d}:{minutes:02d}")

            with col2:
                st.markdown("**Card & Address**")
                st.write(f"Card number: {case['card1']}")
                st.write(f"Card type: {format_value(case['card4'])} / {format_value(case['card6'])}")
                st.write(f"Billing address code: {format_value(case['addr1'])} / {format_value(case['addr2'])}")
                st.write(f"Distance signal: {format_value(case['dist1'])}")

            with col3:
                st.markdown("**Identity & Device**")
                st.write(f"Device type: {format_value(case['devicetype'])}")
                st.write(f"Device info: {format_value(case['deviceinfo'])}")
                st.write(f"Payment email: {format_value(case['p_emaildomain'])}")
                st.write(f"Receiver email: {format_value(case['r_emaildomain'])}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Why this was flagged - behavioral signals:**")

            st.markdown(
                """
                <style>
                .reason-row {
                    display: flex;
                    align-items: center;
                    padding: 8px 0;
                    border-bottom: 1px solid #f0f0f0;
                }
                .reason-label {
                    width: 260px;
                    font-size: 14px;
                    color: #1f2937;
                }
                .reason-value {
                    width: 120px;
                    font-size: 13px;
                    color: #6b7280;
                }
                .reason-bar-track {
                    flex: 1;
                    background: #f3f4f6;
                    border-radius: 4px;
                    height: 10px;
                    margin: 0 12px;
                    position: relative;
                }
                .reason-bar-fill-up {
                    background: #b91c1c;
                    height: 10px;
                    border-radius: 4px;
                }
                .reason-bar-fill-down {
                    background: #001f3f;
                    height: 10px;
                    border-radius: 4px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            feature_row = get_case_features(case["transactionid"])
            shap_values = explainer.shap_values(feature_row)
            contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
            contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

            max_abs_contribution = max(abs(c[1]) for c in contributions)

            for feature_name, contribution in contributions:
                actual_value = feature_row[feature_name].iloc[0]
                display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
                bar_width_pct = int((abs(contribution) / max_abs_contribution) * 100)
                bar_class = "reason-bar-fill-up" if contribution > 0 else "reason-bar-fill-down"
                arrow = "toward FRAUD" if contribution > 0 else "toward normal"

                st.markdown(
                    f"""
                    <div class="reason-row">
                        <div class="reason-label">{display_name}</div>
                        <div class="reason-value">{format_value(actual_value)}</div>
                        <div class="reason-bar-track">
                            <div class="{bar_class}" style="width: {bar_width_pct}%;"></div>
                        </div>
                        <div style="width: 140px; font-size: 12px; color: #6b7280;">{arrow} ({contribution:+.3f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            if btn_col1.button("Allow Transaction", key=f"allow_{case['transactionid']}"):
                save_decision(case["transactionid"], analyst_name, "dismissed")
                st.success("Transaction allowed - marked as false alarm.")
                st.rerun()

            if btn_col2.button("Mark as Fraud", key=f"fraud_{case['transactionid']}"):
                save_decision(case["transactionid"], analyst_name, "confirmed_fraud")
                st.success("Card marked as confirmed fraud.")
                st.rerun()

            if btn_col3.button("Escalate for Review", key=f"escalate_{case['transactionid']}"):
                save_decision(case["transactionid"], analyst_name, "escalated")
                st.warning("Escalated for senior review.")
                st.rerun()



