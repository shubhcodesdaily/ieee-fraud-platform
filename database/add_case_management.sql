-- Table: flagged_cases
-- Transactions the model scored above the decision threshold.
DROP TABLE IF EXISTS analyst_decisions CASCADE;
DROP TABLE IF EXISTS flagged_cases CASCADE;

CREATE TABLE flagged_cases (
    transactionid INT PRIMARY KEY REFERENCES transactions(transactionid),
    fraud_probability NUMERIC(6, 4) NOT NULL,
    requires_second_approval BOOLEAN NOT NULL DEFAULT FALSE,
    flagged_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table: analyst_decisions
-- The audit trail: every human decision made on a flagged case.
CREATE TABLE analyst_decisions (
    id SERIAL PRIMARY KEY,
    transactionid INT NOT NULL REFERENCES flagged_cases(transactionid),
    analyst_name VARCHAR(100) NOT NULL,
    decision VARCHAR(20) NOT NULL CHECK (decision IN (
        'confirmed_fraud', 'dismissed', 'escalated'
    )),
    decided_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_flagged_probability ON flagged_cases(fraud_probability DESC);
