import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "fraud_detection"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


FEATURE_QUERY = """
SELECT
    transactionid,
    card1,
    transactiondt,
    transactionamt,
    isfraud,

    -- Feature 1: average amount this card spent, using only transactions
    -- strictly before this one (never includes the current row).
    AVG(transactionamt) OVER (
        PARTITION BY card1
        ORDER BY transactiondt
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS card_avg_amt_before_this,

    -- Feature 2: how many transactions this card has made before this one.
    -- A sudden spike in this number signals unusual velocity.
    COUNT(*) OVER (
        PARTITION BY card1
        ORDER BY transactiondt
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS card_txn_count_before_this,

    -- Feature 3: seconds since this card's previous transaction.
    -- Fraud often comes in rapid bursts, so a very small gap is a signal.
    transactiondt - LAG(transactiondt) OVER (
        PARTITION BY card1
        ORDER BY transactiondt
    ) AS seconds_since_last_txn

FROM transactions
ORDER BY card1, transactiondt;
"""


def build_features():
    """Run the feature query and return the result as a DataFrame."""
    logger.info("Connecting to database and building features...")
    connection = get_connection()
    df = pd.read_sql(FEATURE_QUERY, connection)
    connection.close()

    logger.info("Built %s rows x %s columns of features", f"{len(df):,}", df.shape[1])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_features()
    print(df.head(10))