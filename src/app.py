import logging
import pandas as pd
import joblib
import shap
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Load the trained model once, when the service starts - not per request.
# This is what makes real-time scoring actually fast.
model = joblib.load("model.joblib")
explainer = shap.TreeExplainer(model)

# The threshold we found earlier by minimizing real dollar cost, not just
# picking 0.5 arbitrarily.
DECISION_THRESHOLD = 0.07

FEATURE_COLUMNS = [
    "card1",
    "transactionamt",
    "has_identity",
    "uid_avg_amt_before_this",
    "uid_txn_count_before_this",
    "seconds_since_last_txn",
    "email_txn_count_before_this",
]

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    """Only the raw facts a caller would actually know about a new transaction."""
    transactionid: int
    card1: int
    addr1: int | None = None
    transactionamt: float
    transactiondt: int
    p_emaildomain: str | None = None


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is alive."""
    return {"status": "ok", "message": "Fraud detection API is running"}

def compute_features_from_db(txn: Transaction):
    """Insert the transaction, then compute its features from real history in Postgres."""
    import psycopg2
    import pandas as pd

    conn = psycopg2.connect(
        host="localhost", port="5433", dbname="fraud_detection",
        user="postgres", password="Atomic123",
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions (transactionid, isfraud, transactiondt, transactionamt,
                                       productcd, card1, addr1, p_emaildomain)
            VALUES (%s, 0, %s, %s, 'W', %s, %s, %s)
            ON CONFLICT (transactionid) DO NOTHING
            """,
            (txn.transactionid, txn.transactiondt, txn.transactionamt,
             txn.card1, txn.addr1, txn.p_emaildomain),
        )
    conn.commit()

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
        WHERE transactionid = {txn.transactionid};
    """
    df = pd.read_sql(query, conn)
    conn.close()

    for col in ["uid_avg_amt_before_this", "uid_txn_count_before_this",
                "seconds_since_last_txn", "email_txn_count_before_this"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[FEATURE_COLUMNS]

@app.post("/score")
def score_transaction(transaction: Transaction):
    row = compute_features_from_db(transaction)

    probability = model.predict_proba(row)[:, 1][0]
    is_flagged = bool(probability >= DECISION_THRESHOLD)

    shap_values = explainer.shap_values(row)
    contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

    reasons = [
        {"feature": name, "contribution": round(float(value), 4)}
        for name, value in contributions
    ]

    return {
        "fraud_probability": round(float(probability), 4),
        "flagged": is_flagged,
        "threshold_used": DECISION_THRESHOLD,
        "reasons": reasons,
    }