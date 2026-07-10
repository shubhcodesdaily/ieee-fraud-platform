import logging
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features import build_features

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "card1",
    "transactionamt",
    "has_identity",
    "uid_avg_amt_before_this",
    "uid_txn_count_before_this",
    "seconds_since_last_txn",
    "email_txn_count_before_this",
]

TARGET_COLUMN = "isfraud"


def split_by_time(df, train_fraction=0.8):
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
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    predicted_probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, predicted_probabilities)
    pr_auc = average_precision_score(y_test, predicted_probabilities)
    logger.info("ROC AUC: %.4f", roc_auc)
    logger.info("PR AUC (average precision): %.4f", pr_auc)
    return roc_auc, pr_auc


def find_best_threshold(model, test_df, false_alarm_cost=10):
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values
    amounts = test_df["transactionamt"].values
    predicted_probabilities = model.predict_proba(X_test)[:, 1]
    best_threshold = 0.5
    best_total_cost = None
    for threshold in [i / 100 for i in range(1, 100)]:
        predicted_fraud = predicted_probabilities >= threshold
        missed_fraud_mask = (y_test == 1) & (~predicted_fraud)
        missed_fraud_cost = amounts[missed_fraud_mask].sum()
        false_alarm_count = ((y_test == 0) & predicted_fraud).sum()
        false_alarm_total_cost = false_alarm_count * false_alarm_cost
        total_cost = missed_fraud_cost + false_alarm_total_cost
        if best_total_cost is None or total_cost < best_total_cost:
            best_total_cost = total_cost
            best_threshold = threshold
    logger.info(
        "Best threshold: %.2f (estimated cost: $%.2f)",
        best_threshold, best_total_cost,
    )
    return best_threshold, best_total_cost


def run_training():
    df = build_features()
    train_df, test_df = split_by_time(df)
    model = train_model(train_df)
    roc_auc, pr_auc = evaluate_model(model, test_df)
    best_threshold, best_cost = find_best_threshold(model, test_df)
    return model, roc_auc, pr_auc, best_threshold, best_cost


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_training()


def explain_one_prediction(model, test_df, row_index=0):
    import shap

    X_test = test_df[FEATURE_COLUMNS]
    one_row = X_test.iloc[[row_index]]

    predicted_probability = model.predict_proba(one_row)[:, 1][0]
    actual_label = test_df[TARGET_COLUMN].iloc[row_index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(one_row)

    logger.info("Transaction (row %s): predicted fraud probability = %.4f (actual label: %s)", row_index, predicted_probability, actual_label)
    logger.info("Feature contributions (positive = pushed toward fraud):")

    contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

    for feature_name, contribution in contributions:
        feature_value = one_row[feature_name].iloc[0]
        logger.info("  %s = %s  ->  contribution: %+.4f", feature_name, feature_value, contribution)

    return contributions


def explain_one_prediction(model, test_df, row_index=0):
    import shap

    X_test = test_df[FEATURE_COLUMNS]
    one_row = X_test.iloc[[row_index]]

    predicted_probability = model.predict_proba(one_row)[:, 1][0]
    actual_label = test_df[TARGET_COLUMN].iloc[row_index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(one_row)

    logger.info("Transaction (row %s): predicted fraud probability = %.4f (actual label: %s)", row_index, predicted_probability, actual_label)
    logger.info("Feature contributions (positive = pushed toward fraud):")

    contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

    for feature_name, contribution in contributions:
        feature_value = one_row[feature_name].iloc[0]
        logger.info("  %s = %s  ->  contribution: %+.4f", feature_name, feature_value, contribution)

    return contributions
