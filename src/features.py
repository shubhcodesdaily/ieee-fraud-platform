"""
Feature engineering: turns raw transaction rows into the behavioral
signals the fraud model actually learns from.

Each feature here traces back to a real fraud-detection idea - this
file is intentionally organized so you can see exactly which raw
column produced which signal, and why that signal matters.
"""

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
    "sslmode": "require",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def build_the_feature_query():
    """
    Build the full SQL query that turns raw transactions into engineered
    features. Written as one big string, but organized section by
    section so each group of features is easy to find and explain.
    """

    # First, we join transactions to identity data, and build a more
    # precise customer identifier (uid) than card1 alone - the same
    # card number can be reissued to a different real person over time,
    # so combining it with the billing address gets us closer to "the
    # same actual customer."
    setup_section = """
        WITH combined AS (
            SELECT
                t.transactionid,
                t.card1,
                t.addr1,
                t.transactiondt,
                t.transactionamt,
                t.isfraud,
                t.p_emaildomain,
                t.d1,
                t.c1,
                t.c2,
                t.m1,
                t.m2,
                t.m3,
                t.m4,
                CONCAT(t.card1, '_', COALESCE(t.addr1::text, 'NA')) AS uid,
                i.devicetype,
                CASE WHEN i.transactionid IS NOT NULL THEN 1 ELSE 0 END AS has_identity
            FROM transactions t
            LEFT JOIN identities i ON t.transactionid = i.transactionid
        )
    """

    # These are the behavioral velocity features we built early on:
    # how this customer's own history compares to the current transaction.
    behavioral_velocity_features = """
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

        COUNT(*) OVER (
            PARTITION BY p_emaildomain ORDER BY transactiondt
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS email_txn_count_before_this
    """

    # NEW: account maturity. D1 is roughly "days since this card was
    # first seen." A brand new card making a large purchase right away
    # is a genuinely well-known fraud pattern.
    account_maturity_feature = """
        COALESCE(d1, 0) AS account_age_days
    """

    # NEW: identity linkage. C1 and C2 are counts of things like
    # addresses or devices linked to this card. We add them together
    # into one combined score - a card suddenly tied to many addresses
    # or devices is a sign of it being passed around or taken over.
    identity_linkage_feature = """
        COALESCE(c1, 0) + COALESCE(c2, 0) AS identity_linkage_score
    """

    # NEW: consistency checks. M1 through M4 are match flags (name
    # matches, address matches, etc). We count how many of them did
    # NOT come back as a clean "T" - a mismatch or a missing check is
    # itself a red flag, so more mismatches means a higher score.
    match_check_feature = """
        (CASE WHEN m1 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) +
        (CASE WHEN m2 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) +
        (CASE WHEN m3 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) +
        (CASE WHEN m4 IS DISTINCT FROM 'T' THEN 1 ELSE 0 END) AS match_flag_count
    """

    # NEW: time-of-day and day-of-week risk. TransactionDT is just a
    # count of seconds elapsed, not a real timestamp - but we can still
    # pull "which hour of the day" and "which day of the week" out of
    # it, since fraud tends to cluster at unusual hours.
    time_pattern_features = """
        MOD(transactiondt / 3600, 24) AS hour_of_day,
        MOD(transactiondt / 86400, 7) AS day_of_week
    """

    # NEW: an interaction feature. A transaction with no identity data
    # AND a very short gap since the last transaction on this card is a
    # stronger signal together than either fact would be on its own.
    interaction_feature = """
        CASE
            WHEN has_identity = 0
                 AND (transactiondt - LAG(transactiondt) OVER (
                        PARTITION BY uid ORDER BY transactiondt
                     )) < 300
            THEN 1 ELSE 0
        END AS no_identity_high_velocity
    """

    full_query = f"""
        {setup_section}
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
            {behavioral_velocity_features},
            {account_maturity_feature},
            {identity_linkage_feature},
            {match_check_feature},
            {time_pattern_features},
            {interaction_feature}
        FROM combined
        ORDER BY uid, transactiondt;
    """

    return full_query


def build_features():
    """Run the feature query and return the result as a DataFrame."""
    logger.info("Connecting to database and building features...")

    connection = get_connection()
    query = build_the_feature_query()
    features_df = pd.read_sql(query, connection)
    connection.close()

    logger.info(
        "Built %s rows x %s columns of features",
        f"{len(features_df):,}", features_df.shape[1],
    )
    return features_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_features()
    print(df.head(10))