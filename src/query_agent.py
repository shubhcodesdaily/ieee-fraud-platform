import logging
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.features import get_connection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]

SCHEMA_DESCRIPTION = """
You have access to a fraud detection PostgreSQL database with these tables:

transactions (transactionid, isfraud, transactiondt, transactionamt, productcd,
    card1, card4, card6, addr1, addr2, dist1, p_emaildomain, r_emaildomain)

identities (transactionid, devicetype, deviceinfo)

flagged_cases (transactionid, fraud_probability, requires_second_approval, flagged_at)

analyst_decisions (id, transactionid, analyst_name, decision, decided_at)
    decision is one of: 'confirmed_fraud', 'dismissed', 'escalated'

activity_log (transactionid, transactionamt, fraud_probability, was_flagged, processed_at)
"""


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


def generate_sql(question):
    llm = get_llm()

    prompt = f"""{SCHEMA_DESCRIPTION}

A fraud analyst asked this question: "{question}"

Write ONE PostgreSQL SELECT query that answers it. Rules:
- Only a SELECT statement - never modify data.
- Always include a LIMIT (max 50 rows) unless the question asks for a count.
- Return ONLY the raw SQL, no explanation, no markdown formatting, no backticks.
"""

    response = llm.invoke(prompt)
    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def is_safe_query(sql):
    upper_sql = sql.upper()
    if not upper_sql.strip().startswith("SELECT"):
        return False
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in upper_sql:
            return False
    return True


def run_query(sql):
    conn = get_connection()
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def summarize_result(question, df):
    llm = get_llm()

    prompt = f"""A fraud analyst asked: "{question}"

The database returned this result:
{df.to_string(index=False, max_rows=20)}

Write a short, plain-English answer (2-3 sentences) summarizing what this
means for the analyst. Be direct and specific with numbers.
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def ask_question(question):
    logger.info("Question: %s", question)

    sql = generate_sql(question)
    logger.info("Generated SQL: %s", sql)

    if not is_safe_query(sql):
        return {
            "sql": sql,
            "data": None,
            "summary": "This question would require modifying data, which is not allowed. Please ask a question that only reads data.",
            "error": True,
        }

    try:
        df = run_query(sql)
    except Exception as exc:
        return {
            "sql": sql,
            "data": None,
            "summary": f"The query failed to run: {exc}",
            "error": True,
        }

    summary = summarize_result(question, df)

    return {
        "sql": sql,
        "data": df,
        "summary": summary,
        "error": False,
    }


if __name__ == "__main__":
    result = ask_question("How many cases are pending review?")
    print("SQL:", result["sql"])
    print("Summary:", result["summary"])
    print(result["data"])