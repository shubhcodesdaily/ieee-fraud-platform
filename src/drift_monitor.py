import logging

import pandas as pd
from scipy.stats import ks_2samp

from src.features import build_features
from src.train import split_by_time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Note: uid_txn_count_before_this and email_txn_count_before_this are
# cumulative counts that mechanically increase over time - they will
# almost always show statistical "drift" between an early and late
# period, even with no real change in fraud behavior. transactionamt
# is the more meaningful signal here, since it should stay stable
# unless spending patterns genuinely shift.
MONITORED_FEATURES = [
    "transactionamt",
    "uid_avg_amt_before_this",
    "uid_txn_count_before_this",
    "seconds_since_last_txn",
    "email_txn_count_before_this",
]

# A p-value below this means the two distributions are significantly
# different - i.e. real drift, not just normal random variation.
DRIFT_P_VALUE_THRESHOLD = 0.05


def check_drift(reference_df, current_df):
    """
    Compare each monitored feature's distribution between a reference
    period (what the model was trained on) and a current period (recent
    data), using the Kolmogorov-Smirnov test.

    Returns a DataFrame: one row per feature, with the test statistic,
    p-value, and whether drift was detected.
    """
    results = []

    for feature in MONITORED_FEATURES:
        reference_values = reference_df[feature].dropna()
        current_values = current_df[feature].dropna()

        if len(reference_values) < 10 or len(current_values) < 10:
            logger.warning("Skipping %s - not enough data in one of the periods", feature)
            continue

        statistic, p_value = ks_2samp(reference_values, current_values)
        drift_detected = p_value < DRIFT_P_VALUE_THRESHOLD

        results.append({
            "feature": feature,
            "ks_statistic": round(statistic, 4),
            "p_value": round(p_value, 6),
            "drift_detected": drift_detected,
        })

    return pd.DataFrame(results)


def run_drift_check(recent_fraction=0.1):
    """
    Split the full dataset by time: everything except the most recent
    slice is "reference" (what the model trained on), and the most
    recent slice is "current" (a stand-in for newly arriving data).

    In a real production system, "current" would be live transactions
    from the last day/week rather than a slice of historical data - but
    this demonstrates the exact same statistical mechanism.
    """
    df = build_features()
    df_sorted = df.sort_values("transactiondt").reset_index(drop=True)

    split_point = int(len(df_sorted) * (1 - recent_fraction))
    reference_df = df_sorted.iloc[:split_point]
    current_df = df_sorted.iloc[split_point:]

    logger.info(
        "Comparing reference period (%s rows) against current period (%s rows)",
        f"{len(reference_df):,}", f"{len(current_df):,}",
    )

    drift_report = check_drift(reference_df, current_df)

    logger.info("=" * 60)
    logger.info("DRIFT REPORT")
    logger.info("=" * 60)
    for _, row in drift_report.iterrows():
        status = "DRIFT DETECTED" if row["drift_detected"] else "stable"
        logger.info(
            "%-30s p=%.6f  ks=%.4f  [%s]",
            row["feature"], row["p_value"], row["ks_statistic"], status,
        )
    logger.info("=" * 60)

    any_drift = drift_report["drift_detected"].any()
    if any_drift:
        logger.warning("ACTION NEEDED: One or more features have drifted. Consider retraining.")
    else:
        logger.info("No significant drift detected. Model is likely still representative.")

    return drift_report


if __name__ == "__main__":
    run_drift_check()