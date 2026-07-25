import logging

import psycopg2

from src.features import DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SQL_FILES = [
    "database/init.sql",
    "database/add_case_management.sql",
    "database/add_activity_log.sql",
]


def run_sql_file(conn, path):
    logger.info("Running %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("  done.")


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        for path in SQL_FILES:
            run_sql_file(conn, path)
    finally:
        conn.close()
    logger.info("Schema setup complete on Neon.")


if __name__ == "__main__":
    main()