import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
RAW_DIR = Path("data")


def read_transactions(sample_rows=None):
    path = RAW_DIR / "train_transaction.csv"
    logger.info("Reading %s", path)
    df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    logger.info("  -> %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return df


def read_identity(sample_rows=None):
    path = RAW_DIR / "train_identity.csv"
    logger.info("Reading %s", path)
    df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    logger.info("  -> %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return df


def ingest(sample_rows=None):
    transactions = read_transactions(sample_rows)
    identity = read_identity(sample_rows)
    return transactions, identity