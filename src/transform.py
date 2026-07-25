"""
Transform step of the ETL pipeline.

This file decides which raw columns we keep from the two Kaggle files,
and joins them into one clean table.

The columns we keep are organized into groups, based on what KIND of
fraud signal they support. This isn't just for tidiness - it matches
how features.py later turns these raw columns into actual engineered
features, and it matches how the dashboard explains WHY a transaction
was flagged (behavioral signal, geographic signal, etc).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_columns_we_keep_from_transactions():
    """
    Build the full list of transaction columns we care about, one
    fraud-signal group at a time. Written as separate lists and then
    combined, so each group is easy to read and easy to extend later.
    """
    identifiers_and_label = ["TransactionID", "isFraud"]

    # GROUP 1: Behavioral velocity
    # Is this customer spending or acting differently than their own
    # normal history? This is the group our engineered features
    # (average spend, time since last transaction) are built from.
    behavioral_velocity_columns = ["TransactionDT", "TransactionAmt"]

    # GROUP 2: Geographic / device consistency
    # Does this transaction happen from where it plausibly should?
    # A large distance between billing and shipping/IP address is a
    # classic sign of a stolen card being used far from its owner.
    geographic_consistency_columns = ["dist1", "dist2"]

    # GROUP 3: Account maturity
    # Is this a brand new card, or one with real history? Kaggle
    # documents D1 as roughly "days since this card was first seen" -
    # new accounts making large purchases are disproportionately risky.
    account_maturity_columns = [f"D{i}" for i in range(1, 16)]

    # GROUP 4: Identity linkage
    # Is this card unusually tangled up with many other addresses or
    # devices? Vesta's own team confirmed these are counts like "how
    # many addresses/devices/IPs are linked to this card" - a sudden
    # jump here can signal a card being passed around or reused.
    identity_linkage_columns = [f"C{i}" for i in range(1, 15)]

    # GROUP 5: Consistency / match checks
    # Do the stated identity details actually agree with each other?
    # For example: does the name on the card match the shipping name,
    # does the phone area code match the billing zip code.
    match_check_columns = [f"M{i}" for i in range(1, 10)]

    # GROUP 6: Transaction context
    # What kind of purchase is this? Different product categories and
    # card types carry different baseline fraud risk.
    transaction_context_columns = ["ProductCD", "card4", "card6"]

    # Extra card and address columns we keep for the UID and for
    # display in the analyst dashboard, even though we don't build
    # dedicated engineered features from every one of them yet.
    supporting_card_and_address_columns = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

    # GROUP 7: Email behavior
    # Is the email pattern unusual? A rare or newly-active email
    # domain, or a mismatch between purchaser and recipient email,
    # can signal synthetic identity fraud.
    email_behavior_columns = ["P_emaildomain", "R_emaildomain"]

    all_columns = (
        identifiers_and_label
        + behavioral_velocity_columns
        + geographic_consistency_columns
        + account_maturity_columns
        + identity_linkage_columns
        + match_check_columns
        + transaction_context_columns
        + supporting_card_and_address_columns
        + email_behavior_columns
    )

    return all_columns


def get_columns_we_keep_from_identity():
    """
    From the identity file, we keep the TransactionID (to join on),
    the first 12 numeric id_ columns, and the device information.
    The identity file has up to id_38, but the later columns are
    mostly categorical network/browser fingerprint data we don't use.
    """
    join_key = ["TransactionID"]
    numeric_identity_signals = [f"id_{i:02d}" for i in range(1, 13)]
    device_columns = ["DeviceType", "DeviceInfo"]

    return join_key + numeric_identity_signals + device_columns


def keep_only_the_columns_we_want(df, columns_we_want):
    """
    Filter a DataFrame down to just the columns we actually want,
    safely skipping any that happen to be missing rather than crashing.
    """
    columns_that_actually_exist = []

    for column_name in columns_we_want:
        if column_name in df.columns:
            columns_that_actually_exist.append(column_name)

    return df[columns_that_actually_exist]


def transform(transactions, identity):
    """
    Run the full transform: filter both tables down to the columns we
    care about, then join them into one table using TransactionID.

    We use a LEFT join, which means every transaction is kept, even if
    it has no matching identity record - most transactions don't, and
    that's normal, not an error.
    """
    wanted_transaction_columns = get_columns_we_keep_from_transactions()
    wanted_identity_columns = get_columns_we_keep_from_identity()

    transactions = keep_only_the_columns_we_want(transactions, wanted_transaction_columns)
    identity = keep_only_the_columns_we_want(identity, wanted_identity_columns)

    logger.info(
        "Merging %s transactions with %s identity rows",
        f"{len(transactions):,}", f"{len(identity):,}",
    )

    combined_df = transactions.merge(identity, on="TransactionID", how="left")

    # Convert pandas' internal "missing value" marker into plain None,
    # since that's what our database expects for empty values.
    combined_df = combined_df.where(pd.notnull(combined_df), None)

    return combined_df

TRANSACTION_COLUMNS = get_columns_we_keep_from_transactions()
IDENTITY_COLUMNS = get_columns_we_keep_from_identity()