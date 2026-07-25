-- database/init.sql
--
-- This file defines the two core tables for the fraud detection project.
-- Run it any time you need a fresh copy of the schema # it drops and
-- recreates everything, so it's always safe to rerun.

DROP TABLE IF EXISTS identities CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;

CREATE TABLE transactions (

    -- Every transaction has a unique ID and a fraud label (0 or 1).
    -- The label comes from real investigations, not a guess.
    TransactionID INT PRIMARY KEY,
    isFraud INT NOT NULL CHECK (isFraud IN (0, 1)),

    -- TransactionDT is not a real date # it's a count of seconds since
    -- some reference point Kaggle never disclosed. We treat it purely
    -- as a way to order transactions in time, nothing more.
    TransactionDT INT NOT NULL,
    TransactionAmt NUMERIC(10, 2) NOT NULL,
    ProductCD VARCHAR(10) NOT NULL,

    -- Card and address details. card2, card3, card5, addr1, addr2, and
    -- the distance fields are stored as REAL (not INT) because they
    -- genuinely contain missing values for some transactions # a plain
    -- integer column can't represent "this value is blank."
    card1 INT,
    card2 REAL,
    card3 REAL,
    card4 VARCHAR(25),
    card5 REAL,
    card6 VARCHAR(25),
    addr1 REAL,
    addr2 REAL,
    dist1 REAL,
    dist2 REAL,
    P_emaildomain VARCHAR(100),
    R_emaildomain VARCHAR(100),

    -- D1 through D15: time-gap features, such as days since this card
    -- was first seen, or days since its last transaction. Kaggle only
    -- confirmed the general idea, not what each specific column means.
    D1 REAL,
    D2 REAL,
    D3 REAL,
    D4 REAL,
    D5 REAL,
    D6 REAL,
    D7 REAL,
    D8 REAL,
    D9 REAL,
    D10 REAL,
    D11 REAL,
    D12 REAL,
    D13 REAL,
    D14 REAL,
    D15 REAL,

    -- C1 through C14: counting features # roughly, how many addresses,
    -- devices, or IPs are linked to this card. Again, the exact meaning
    -- of each individual column was never disclosed.
    C1 REAL,
    C2 REAL,
    C3 REAL,
    C4 REAL,
    C5 REAL,
    C6 REAL,
    C7 REAL,
    C8 REAL,
    C9 REAL,
    C10 REAL,
    C11 REAL,
    C12 REAL,
    C13 REAL,
    C14 REAL,
    M1 VARCHAR(10),
    M2 VARCHAR(10), 
    M3 VARCHAR(10), 
    M4 VARCHAR(10),
    M5 VARCHAR(10), 
    M6 VARCHAR(10), 
    M7 VARCHAR(10), 
    M8 VARCHAR(10), 
    M9 VARCHAR(10)
);  

-- The identity table only covers a subset of transactions (about a
-- quarter of them, in the real dataset). That's expected, not an error #
-- most transactions simply have no identity record at all.
CREATE TABLE identities (
    TransactionID INT PRIMARY KEY,
    id_01 REAL,
    id_02 REAL,
    id_03 REAL,
    id_04 REAL,
    id_05 REAL,
    id_06 REAL,
    id_07 REAL,
    id_08 REAL,
    id_09 REAL,
    id_10 REAL,
    id_11 REAL,
    id_12 VARCHAR(20),
    DeviceType VARCHAR(50),
    DeviceInfo VARCHAR(255),
    CONSTRAINT fk_transaction
        FOREIGN KEY (TransactionID)
        REFERENCES transactions(TransactionID)
        ON DELETE CASCADE
);

-- Indexes on the two columns we'll actually filter by at query time #
-- indexing everything would be wasteful, since most columns are never
-- searched on directly.
CREATE INDEX idx_trans_dt ON transactions(TransactionDT);
CREATE INDEX idx_trans_card1 ON transactions(card1);