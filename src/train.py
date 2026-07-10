import logging

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features import build_features

logger = logging.getLogger(__name__)

# The columns the model is allowed to learn from. We deliberately exclude
# transactionid (just a row identifier, no predictive meaning) and isfraud
# (that's the answer we're trying to predict, not a clue).
FEATURE_COLUMNS = [
    "card1",
    "transactionamt",
    "card_avg_amt_before_this",
    "card_txn_count_before_this",
    "seconds_since_last_txn",
]

TARGET_COLUMN = "isfraud"


def split_by_time(df, train_fraction=0.8):
    """
    Split into train/test using TIME, not randomly.

    We sort every row by transactiondt (the real time order), then take the
    first 80% as training data and the last 20% as test data. This mimics
    reality: a fraud model can only ever learn from the past, then must
    predict transactions that haven't happened yet.

    A random split would let the model "see the future" during training,
    which gives a fake, inflated score that would fall apart in production.
    """
    df_sorted = df.sort_values("transactiondt").reset_index(drop=True)

    split_point = int(len(df_sorted) * train_fraction)

    train_df = df_sorted.iloc[:split_point]
    test_df = df_sorted.iloc[split_point:]

    logger.info(
        "Time split: %s train rows (up to t=%s), %s test rows (from t=%s)",
        f"{len(train_df):,}", train_df["transactiondt"].max(),
        f"{len(test_df):,}", test_df["transactiondt"].min(),
    )

    return train_df, test_df


def train_model(train_df):
    """
    Train a LightGBM classifier on the training split.

    LightGBM handles missing values (NaN) natively, which matters a lot
    here: our engineered features are NaN whenever a card has no prior
    history, and that missingness is itself meaningful, not noise to be
    filled in.
    """
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    logger.info("Training LightGBM on %s rows...", f"{len(X_train):,}")

    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        n_estimators=200,
    )
    model.fit(X_train, y_train)

    logger.info("Training complete.")
    return model


def evaluate_model(model, test_df):
    """
    Score the model on the held-out (future) test set and report two
    metrics that actually matter for a rare-event problem like fraud:

    - ROC AUC: how well the model ranks fraud above non-fraud overall.
    - PR AUC (average precision): more honest for imbalanced data, since
      only ~3.5% of transactions are fraud. A model that just guessed
      "never fraud" would score ~96.5% accuracy but be useless - PR AUC
      does not reward that kind of laziness.
    """
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    predicted_probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, predicted_probabilities)
    pr_auc = average_precision_score(y_test, predicted_probabilities)

    logger.info("ROC AUC: %.4f", roc_auc)
    logger.info("PR AUC (average precision): %.4f", pr_auc)

    return roc_auc, pr_auc


def run_training():
    df = build_features()
    train_df, test_df = split_by_time(df)
    model = train_model(train_df)
    roc_auc, pr_auc = evaluate_model(model, test_df)
    return model, roc_auc, pr_auc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_training()