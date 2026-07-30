import logging
import random
import time

import joblib
import pandas as pd

from src.features import get_connection
from src.train import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.07
HIGH_VALUE_THRESHOLD = 500


def generate_new_transaction(next_id, current_max_dt):
    """
    Build one new, realistic-looking transaction - as if it just happened
    right now. Reuses an existing real card1/addr1 sometimes (so it has
    real history to compare against) and sometimes a brand new one (so we
    see the "no history yet" case too).
    """
    is_returning_card = random.random() < 0.7

    conn = get_connection()
    with conn.cursor() as cur:
        if is_returning_card:
            cur.execute("SELECT card1, addr1 FROM transactions ORDER BY RANDOM() LIMIT 1;")
            card1, addr1 = cur.fetchone()
        else:
            card1 = random.randint(1000, 18000)
            addr1 = random.randint(100, 540)
    conn.close()

    is_suspicious = random.random() < 0.15
    amount = round(random.uniform(500, 2000), 2) if is_suspicious else round(random.uniform(10, 150), 2)

    return {
        "transactionid": next_id,
        "isfraud": 0,
        "transactiondt": current_max_dt + random.randint(30, 600),
        "transactionamt": amount,
        "productcd": random.choice(["W", "H", "C", "S", "R"]),
        "card1": card1,
        "card4": random.choice(["visa", "mastercard", "amex", "discover"]),
        "card6": random.choice(["debit", "credit"]),
        "addr1": addr1,
        "p_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"]),
    }


def insert_transaction(txn):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions
                (transactionid, isfraud, transactiondt, transactionamt,
                 productcd, card1, card4, card6, addr1, p_emaildomain)
            VALUES (%(transactionid)s, %(isfraud)s, %(transactiondt)s, %(transactionamt)s,
                    %(productcd)s, %(card1)s, %(card4)s, %(card6)s, %(addr1)s, %(p_emaildomain)s)
            """,
            txn,
        )
    conn.commit()
    conn.close()


def compute_features_for_one(transaction_id):
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


def flag_if_needed(transaction_id, probability, amount):
    if probability < DECISION_THRESHOLD:
        return False
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO flagged_cases (transactionid, fraud_probability, requires_second_approval) VALUES (%s, %s, %s)",
            (transaction_id, float(probability), amount >= HIGH_VALUE_THRESHOLD),
        )
    conn.commit()
    conn.close()
    return True


def run_live_feed(delay_seconds=3):
    model = joblib.load("model.joblib")

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(transactionid) FROM transactions;")
        next_id = cur.fetchone()[0] + 1
        cur.execute("SELECT MAX(transactiondt) FROM transactions;")
        current_max_dt = cur.fetchone()[0]
    conn.close()

    logger.info("Starting live feed. Press Ctrl+C to stop.")

    while True:
        txn = generate_new_transaction(next_id, current_max_dt)
        insert_transaction(txn)

        features = compute_features_for_one(txn["transactionid"])
        probability = model.predict_proba(features)[:, 1][0]

        flagged = flag_if_needed(txn["transactionid"], probability, txn["transactionamt"])
        status = "FLAGGED" if flagged else "ok"

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO activity_log (transactionid, transactionamt, fraud_probability, was_flagged) VALUES (%s, %s, %s, %s)",
                (txn["transactionid"], txn["transactionamt"], float(probability), flagged),
            )
        conn.commit()
        conn.close()

        logger.info(
            "Txn %s | £%.2f | risk=%.1f%% | %s",
            txn["transactionid"], txn["transactionamt"], probability * 100, status,
        )

        next_id += 1
        current_max_dt = txn["transactiondt"]
        time.sleep(delay_seconds)


if __name__ == "__main__":
    run_live_feed()