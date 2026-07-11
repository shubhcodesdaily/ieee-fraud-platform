import logging

import joblib
import psycopg2
import psycopg2.extras

from src.features import build_features, get_connection
from src.train import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.07

# Transactions above this amount need a second analyst to approve,
# mirroring the "four-eyes" review pattern used at real banks for
# high-value or high-risk decisions.
HIGH_VALUE_THRESHOLD = 500


def flag_cases():
    model = joblib.load("model.joblib")
    df = build_features()

    X = df[FEATURE_COLUMNS]
    df["fraud_probability"] = model.predict_proba(X)[:, 1]

    flagged = df[df["fraud_probability"] >= DECISION_THRESHOLD].copy()
    flagged["requires_second_approval"] = flagged["transactionamt"] >= HIGH_VALUE_THRESHOLD

    logger.info("Flagging %s of %s transactions (threshold=%.2f)", f"{len(flagged):,}", f"{len(df):,}", DECISION_THRESHOLD)

    rows = list(flagged[["transactionid", "fraud_probability", "requires_second_approval"]].itertuples(index=False, name=None))

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE analyst_decisions, flagged_cases RESTART IDENTITY CASCADE;")
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO flagged_cases (transactionid, fraud_probability, requires_second_approval) VALUES %s",
            rows,
        )
    conn.commit()
    conn.close()

    logger.info("Done. %s cases loaded into flagged_cases.", f"{len(rows):,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flag_cases()
