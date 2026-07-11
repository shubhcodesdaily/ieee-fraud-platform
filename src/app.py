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
    """
    The shape of a transaction the API expects to receive.
    Pydantic automatically validates types and rejects bad requests -
    e.g. if card1 is sent as text instead of a number, this catches it
    before our code ever runs.
    """
    card1: int
    transactionamt: float
    has_identity: int
    uid_avg_amt_before_this: float | None = None
    uid_txn_count_before_this: int = 0
    seconds_since_last_txn: float | None = None
    email_txn_count_before_this: int = 0


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is alive."""
    return {"status": "ok", "message": "Fraud detection API is running"}


@app.post("/score")
def score_transaction(transaction: Transaction):
    """
    Score one transaction and return a decision plus the reasons why.
    """
    # Convert the incoming transaction into the same column order the
    # model was trained on - this consistency is what prevents
    # "training/serving skew".
    row = pd.DataFrame([{
        "card1": transaction.card1,
        "transactionamt": transaction.transactionamt,
        "has_identity": transaction.has_identity,
        "uid_avg_amt_before_this": transaction.uid_avg_amt_before_this,
        "uid_txn_count_before_this": transaction.uid_txn_count_before_this,
        "seconds_since_last_txn": transaction.seconds_since_last_txn,
        "email_txn_count_before_this": transaction.email_txn_count_before_this,
    }])

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