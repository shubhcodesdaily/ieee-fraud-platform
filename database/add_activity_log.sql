DROP TABLE IF EXISTS activity_log CASCADE;
CREATE TABLE activity_log (
    id SERIAL PRIMARY KEY,
    transactionid INT NOT NULL,
    transactionamt NUMERIC(10, 2) NOT NULL,
    fraud_probability NUMERIC(6, 4) NOT NULL,
    was_flagged BOOLEAN NOT NULL,
    processed_at TIMESTAMP NOT NULL DEFAULT NOW()
);
