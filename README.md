# Sentinel Fraud Intelligence
### An End-to-End Fraud Detection Platform — IEEE-CIS Dataset

  <img width="747" height="107" alt="image" src="https://github.com/user-attachments/assets/ce8e79cd-f21b-4289-addf-76955708daeb" />


## Tech stack

Python · PostgreSQL (Neon) · Docker · SQL window functions · pandas · LightGBM · scikit-learn · SHAP · FastAPI · Streamlit · MLflow · Power BI · Git/GitHub

---

## Why I built this

Fraud detection is a genuinely hard, high-stakes problem: roughly 3.5% of transactions in real payment data are fraudulent, and a system that gets this wrong in either direction is expensive — miss a fraud and you lose real money; wrongly block a legitimate customer and you lose their trust. I wanted to build something that went beyond "train a model, report an accuracy score" — I wanted to build the *system* a real fraud team would actually use: a pipeline that engineers meaningful behavioral signals, a model whose decisions can be explained in plain English, a cost-aware decision threshold grounded in real dollars, and a human-in-the-loop workflow with a genuine audit trail — because that's how real fintechs (Monzo, Revolut) are documented to actually operate, and it's what regulators require.

This project is also, honestly, a demonstration of *engineering judgment under real constraints* — not just a finished, polished result. Along the way I hit and resolved real infrastructure problems (a Docker networking failure, cloud database migration, dependency conflicts), made deliberate trade-offs (excluding 339 anonymized columns to preserve explainability, at a real cost to raw predictive power), and can point to a genuine, current limitation in my own code that I know exactly how I'd fix.

## How I built it

**Data foundation.** The IEEE-CIS dataset (590,540 real transactions, 144,233 identity records) was ingested through a custom ETL pipeline into PostgreSQL — initially in Docker locally, later migrated to Neon (a serverless Postgres platform recently acquired by Databricks) for a genuinely live, cloud-hosted database.

**Feature engineering.** Rather than using all 394 raw columns — most of which (`V1`-`V339`) are anonymized by Kaggle/Vesta with no disclosed meaning — I deliberately restricted the model to columns with documented or well-evidenced meaning, and engineered 13 features across seven signal groups using SQL window functions (see Architecture below). This was a conscious trade-off: it caps my achievable accuracy below what teams using all 394 columns reach, in exchange for a model whose every decision can be explained in plain English to a compliance analyst.

**Modeling.** A LightGBM classifier, evaluated with a **time-based train/test split** (never random - a model can only ever learn from the past to predict the future, and a random split would leak future information into training). Evaluated on PR AUC, not accuracy, since accuracy is meaningless on a 96.5%-legitimate dataset.

**Decisioning.** Instead of an arbitrary 0.5 cutoff, I calculated the actual dollar cost of two kinds of mistakes - missing real fraud vs. wrongly blocking a legitimate transaction - and chose the threshold that minimizes total cost.

**Explainability & governance.** SHAP values explain every prediction. A Streamlit dashboard shows a prioritized case queue where a human analyst - never the algorithm alone - makes the final call, and every decision is permanently logged with a full audit trail.

**MLOps.** MLflow tracks every training run's metrics and parameters. A drift-monitoring module (Kolmogorov-Smirnov test) statistically checks whether new data has diverged from what the model was trained on.

## What I gained from building it

A working, end-to-end system I understand completely, start to finish - not a black box I can only partially explain. A real, quantified engineering result (detailed below) I can defend under questioning. Direct experience with the unglamorous but genuinely important parts of ML engineering: schema migrations, cloud deployment, dependency conflicts, and - a real, humbling lesson - diagnosing and fixing a stubborn Docker networking failure that took a full troubleshooting cycle (reinstall, factory reset, WSL distro rebuild) to resolve. And a clear, honest view of my own project's remaining weaknesses, which I consider as valuable as its strengths.

---

## Screenshots

**Analyst login**

<img width="1906" height="857" alt="image" src="https://github.com/user-attachments/assets/e61eec40-b6bc-46d8-9ef5-24474f70f0a9" />


**Dashboard overview**

<img width="1893" height="994" alt="image" src="https://github.com/user-attachments/assets/03bd4b1a-0ae9-4d6d-b44f-e38760c80854" />


**Case explainability**

`<img width="1187" height="770" alt="image" src="https://github.com/user-attachments/assets/3f92308d-2cdc-4095-b9f0-f6e9a3750c19" />


**Fraud Detection Simulator**

<img width="1147" height="593" alt="image" src="https://github.com/user-attachments/assets/b41b6196-322d-4a4b-a2c5-d1e58faa1ef9" />


**Live Activity Feed**
`[screenshot: funnel summary + color-coded activity cards]`

**Real-time scoring API**
`<img width="1565" height="172" alt="image" src="https://github.com/user-attachments/assets/0b147409-0676-4f28-b177-085c5d7c633e" />
`

**MLflow experiment tracking**
<img width="1916" height="920" alt="image" src="https://github.com/user-attachments/assets/fdf3d66d-7365-4551-b32a-ac3b340873a6" />


---

## Architecture

```
Raw Kaggle CSVs (590,540 transactions, 144,233 identity records)
        |
        v
   ETL (extract -> transform -> load)
        |
        v
  PostgreSQL / Neon  <----------------------------+
  (transactions, identities, flagged_cases,        |
   analyst_decisions, activity_log)                |
        |                                          |
        v                                          |
Feature Engineering (SQL window functions, 13 features)
        |
        v
LightGBM Model (time-based split) -- MLflow tracking
        |
        v
Cost-sensitive threshold + SHAP explainer -- Drift monitor
        |
        +---------------+--------------------------+
        v               v
   FastAPI            Streamlit Dashboard
 (real-time            (case queue, SHAP reasons,
  scoring API)          Fraud Detection Simulator,
                        decisions -> audit trail)
```

**A known, honest limitation:** the feature-computation logic above is currently implemented independently in four places (`features.py`, `app.py`, `live_feed.py`, `dashboard.py`) rather than as one shared function - a natural consequence of iterative development. The correct refactor is a single function in `features.py` that the other three import, eliminating the risk of the four copies silently drifting out of sync. I know exactly how I'd implement this; I prioritized feature breadth over this refactor given project timeline.

---

## The seven feature groups

| Group | Signal | Source columns |
|---|---|---|
| 1. Behavioral velocity | Is this customer spending/acting differently than their own history? | `TransactionDT`, `TransactionAmt` |
| 2. Geographic/device consistency | Does this transaction happen from where it plausibly should? | `dist1`, `dist2` |
| 3. Account maturity | Is this a brand-new card or an established one? | `D1` |
| 4. Identity linkage | Is this card unusually tangled up with many addresses/devices? | `C1`, `C2` |
| 5. Consistency / match checks | Do stated identity details actually agree with each other? | `M1`-`M4` |
| 6. Transaction context | What kind of purchase is this? | `ProductCD`, `card4`, `card6` |
| 7. Email behavior | Is the email pattern unusual? | `P_emaildomain`, `R_emaildomain` |

Column meanings for groups 3-5 are only partially documented by Kaggle/Vesta ("counting, such as how many addresses are found to be associated with the payment card... the actual meaning is masked") - confirmed directly by a Vesta team member on Kaggle's official competition forum. Where the exact definition of an individual column isn't disclosed, this README states that plainly rather than presenting inference as fact.

---

## Results (real, measured, logged in MLflow)

| Model version | Features | ROC AUC | PR AUC | Notes |
|---|---|---|---|---|
| Baseline | `card1` only | 0.777 | 0.166 | Initial working model |
| + UID/identity/email | 7 features | 0.782 | 0.163 | UID construction (`card1`+`addr1`) |
| + account age, identity linkage, match checks | **13 features** | **0.845** | **0.361** | Major improvement |

Adding account-maturity, identity-linkage, and consistency-check features - all directly traceable to Kaggle's own documented column groups - **more than doubled PR AUC** (0.163 -> 0.361), a 121% relative improvement, and reduced estimated business cost on the test set by over $83,000.

---

## Feature list (product features, not ML features)

- Full ETL pipeline against the real 590,540-row dataset
- SQL window-function feature engineering (13 features, 7 signal groups)
- LightGBM model with honest time-based evaluation
- Cost-sensitive decision threshold (minimizes real dollar loss, not accuracy)
- SHAP explainability on every prediction, with plain-English feature labels
- Password-protected analyst dashboard (Streamlit) with:
  - KPI overview (cases in queue, value at risk, pending review)
  - Search and multi-criteria filtering
  - Full case context (transaction, card/address, identity/device)
  - Visual SHAP breakdown per case
  - "Four-eyes" high-value case flagging
  - Decision buttons (Allow / Mark as Fraud / Escalate) writing to a permanent audit trail
  - Live Activity Feed with funnel-style summary (Seen / Flagged / Cleared)
  - **Fraud Detection Simulator** - generate a synthetic transaction with a real date/time picker and watch the model score it live (explicitly labeled as using no real customer data)
- FastAPI real-time scoring service (`/score` endpoint), independently verified working
- MLflow experiment tracking across three real training iterations
- Drift monitoring via Kolmogorov-Smirnov statistical test
- Power BI executive report (aggregate KPIs, decision breakdown, fraud-rate trend)
- Cloud deployment on Neon (PostgreSQL)

---

## Honest limitations

- 339 anonymized Vesta columns (`V1`-`V339`) are deliberately excluded to preserve explainability - this caps achievable accuracy below what an all-columns model would reach
- Feature-computation logic is duplicated across four files rather than centralized (see Architecture note above)
- FastAPI service is verified working when run locally (see screenshot above); not currently deployed as a standalone public endpoint
- The Fraud Detection Simulator uses entirely synthetic input data - no real customer information is used or displayed
- `hour_of_day` / `day_of_week` features are derived from an anonymized time offset, not a real calendar timestamp - genuinely useful to the model, but not human-interpretable as an actual time of day

## Roadmap

- Consolidate duplicated feature-computation logic into one shared function
- Deploy `app.py` as a standalone public API, called by the dashboard over HTTP rather than duplicated in-process
- Redis-backed sub-second velocity features (rolling spend windows, IP/country mismatch detection)
- LangGraph-based natural-language query agent for ad-hoc analyst questions
- LLM-drafted Suspicious Activity Report (SAR) generation for confirmed fraud cases

---

## Tech stack

Python · PostgreSQL (Neon) · Docker · SQL window functions · pandas · LightGBM · scikit-learn · SHAP · FastAPI · Streamlit · MLflow · Power BI · Git/GitHub
