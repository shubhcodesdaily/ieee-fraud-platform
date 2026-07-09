"""Load: write the transformed pipeline output into Postgres per database/init.sql."""

import logging
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from src.transform import IDENTITY_COLUMNS, TRANSACTION_COLUMNS

logger = logging.getLogger(__name__)

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "fraud_detection"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}


def _insert(conn, table, columns, rows):
    if not rows:
        logger.warning("No rows to load into %s", table)
        return
    insert_sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
        f"ON CONFLICT (TransactionID) DO NOTHING"
    )
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, rows, page_size=1000)
    conn.commit()
    logger.info("Loaded %s rows into %s", f"{len(rows):,}", table)


def load(clean_df):
    """Split the merged dataframe back into transactions/identities and load into Postgres."""
    transaction_cols = [c for c in TRANSACTION_COLUMNS if c in clean_df.columns]
    identity_cols = [c for c in IDENTITY_COLUMNS if c in clean_df.columns]
    identity_value_cols = [c for c in identity_cols if c != "TransactionID"]

    transaction_rows = list(clean_df[transaction_cols].itertuples(index=False, name=None))

    identity_df = clean_df[identity_cols].dropna(subset=identity_value_cols, how="all")
    identity_rows = list(identity_df.itertuples(index=False, name=None))

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        _insert(conn, "transactions", transaction_cols, transaction_rows)
        _insert(conn, "identities", identity_cols, identity_rows)
    finally:
        conn.close()
