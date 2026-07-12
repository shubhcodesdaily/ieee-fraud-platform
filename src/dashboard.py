import logging
import joblib
import pandas as pd
import psycopg2
import psycopg2.extras
import shap
import streamlit as st

from src.features import get_connection
from src.train import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(message)s")

st.set_page_config(page_title="Fraud Analyst Dashboard", layout="wide")


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
               seconds_since_last_txn, email_txn_count_before_this
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
                ) AS email_txn_count_before_this
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


def format_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return value


# --- UI starts here ---------------------------------------------------

st.title("Fraud Analyst Dashboard")

model, explainer = load_model()
queue = get_queue()

st.write(f"**{len(queue)} cases** awaiting review, sorted by risk.")

total_value_at_risk = queue["transactionamt"].sum()
avg_risk = queue["fraud_probability"].mean() if len(queue) > 0 else 0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Cases in queue", f"{len(queue):,}")
kpi2.metric("Value at risk", f"GBP {total_value_at_risk:,.2f}")
kpi3.metric("Average risk score", f"{avg_risk:.1%}")

min_risk = st.slider("Minimum risk to display", 0.0, 1.0, 0.0, 0.01)
queue = queue[queue["fraud_probability"] >= min_risk]

with st.sidebar:
    st.subheader("Live Activity Feed")
    if st.button("Refresh feed"):
        st.rerun()
    activity = get_recent_activity()
    for _, row in activity.iterrows():
        tag = "[FLAGGED]" if row["was_flagged"] else "[ok]"
        st.write(f"{tag} Txn {row['transactionid']} - GBP {row['transactionamt']:.2f} - {row['fraud_probability']:.1%}")

analyst_name = st.text_input("Analyst name", value="Shubh")

for _, case in queue.iterrows():
    approval_tag = " [SECOND APPROVAL REQUIRED]" if case["requires_second_approval"] else ""
    header = f"Txn {case['transactionid']} - {case['fraud_probability']:.1%} risk - GBP {case['transactionamt']:.2f}{approval_tag}"

    with st.expander(header):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Transaction**")
            st.write(f"Amount: GBP {case['transactionamt']:.2f}")
            st.write(f"Product category: {format_value(case['productcd'])}")
            st.write(f"Time offset: {case['transactiondt']}")

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

        st.markdown("---")
        st.markdown("**Why this was flagged - behavioral signals:**")

        feature_row = get_case_features(case["transactionid"])
        shap_values = explainer.shap_values(feature_row)
        contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

        for feature_name, contribution in contributions:
            actual_value = feature_row[feature_name].iloc[0]
            direction = "[UP] toward fraud" if contribution > 0 else "[DOWN] toward normal"
            st.write(f"- **{feature_name}** = {format_value(actual_value)} -> {direction} ({contribution:+.3f})")

        st.markdown("---")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        if btn_col1.button("Confirm Fraud", key=f"confirm_{case['transactionid']}"):
            save_decision(case["transactionid"], analyst_name, "confirmed_fraud")
            st.success("Marked as confirmed fraud.")
            st.rerun()

        if btn_col2.button("Dismiss (False Alarm)", key=f"dismiss_{case['transactionid']}"):
            save_decision(case["transactionid"], analyst_name, "dismissed")
            st.success("Dismissed.")
            st.rerun()

        if btn_col3.button("Escalate", key=f"escalate_{case['transactionid']}"):
            save_decision(case["transactionid"], analyst_name, "escalated")
            st.warning("Escalated for further review.")
            st.rerun()