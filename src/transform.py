"""Transform: merge ingested transaction/identity data and shape it to match database/init.sql."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TRANSACTION_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
]

IDENTITY_COLUMNS = [
    "TransactionID",
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
    "id_07", "id_08", "id_09", "id_10", "id_11", "id_12",
    "DeviceType", "DeviceInfo",
]


def transform(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions[[c for c in TRANSACTION_COLUMNS if c in transactions.columns]]
    identity = identity[[c for c in IDENTITY_COLUMNS if c in identity.columns]]

    logger.info("Merging %s transactions with %s identity rows", f"{len(transactions):,}", f"{len(identity):,}")
    clean_df = transactions.merge(identity, on="TransactionID", how="left")
    return clean_df.where(pd.notnull(clean_df), None)
