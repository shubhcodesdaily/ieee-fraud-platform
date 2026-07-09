import logging

from src.ingestion import ingest
from src.transform import transform
from src.load import load

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_pipeline(sample_rows=None):
    transactions, identity = ingest(sample_rows=sample_rows)
    clean_df = transform(transactions, identity)
    load(clean_df)
    return clean_df


if __name__ == "__main__":
    df = run_pipeline(sample_rows=50000)
    print("Done:", df.shape)