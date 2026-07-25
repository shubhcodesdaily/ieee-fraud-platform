import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "ep-solitary-shape-ay0p85by-pooler.c-5.us-east-2.aws.neon.tech"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "neondb"),
    "user": os.getenv("DB_USER", "neondb_owner"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": "require"
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


FEATURE_QUERY = """
WITH combined AS (
    SELECT
        t.transactionid,
        t.card1,
        t.addr1,
        t.transactiondt,
        t.transactionamt,
        t.isfraud,
        t.p_emaildomain,
        -- A more precise identity than card1 alone: combine card + billing
        -- address. The same card1 can be reissued to different people over
        -- time, so this approximates "the same real customer" better.
        CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA')) AS uid,
        i.devicetype,
        -- Flag: does this transaction have any identity record at all?
        -- Missing identity is itself a signal, not just an absence of data.
        CASE WHEN i.transactionid IS NOT NULL THEN 1 ELSE 0 END AS has_identity
    FROM transactions t
    LEFT JOIN identities i ON t.transactionid = i.transactionid
)
SELECT
    transactionid,
    card1,
    uid,
    transactiondt,
    transactionamt,
    isfraud,
    devicetype,
    has_identity,
    p_emaildomain,

    -- Same three behavioral features as before, but now partitioned by the
    -- more precise uid instead of card1 alone.
    AVG(transactionamt) OVER (
        PARTITION BY uid ORDER BY transactiondt
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS uid_avg_amt_before_this,

    COUNT(*) OVER (
        PARTITION BY uid ORDER BY transactiondt
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS uid_txn_count_before_this,

    transactiondt - LAG(transactiondt) OVER (
        PARTITION BY uid ORDER BY transactiondt
    ) AS seconds_since_last_txn,

    -- New: how many prior transactions has this email domain been involved
    -- in? Unusual or rare domains are a classic fraud signal.
    COUNT(*) OVER (
        PARTITION BY p_emaildomain ORDER BY transactiondt
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS email_txn_count_before_this

FROM combined
ORDER BY uid, transactiondt;
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