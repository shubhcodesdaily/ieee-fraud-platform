-- database/init.sql

-- 1. Clean the slate so this script can be run over and over safely
DROP TABLE IF EXISTS identities CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;

-- 2. Create the raw staging table for Transactions
CREATE TABLE transactions (
    TransactionID INT PRIMARY KEY,
    isFraud INT NOT NULL CHECK (isFraud IN (0, 1)),
    TransactionDT INT NOT NULL,  
    TransactionAmt NUMERIC(10, 2) NOT NULL,
    ProductCD VARCHAR(10) NOT NULL,
    card1 INT, card2 REAL, card3 REAL, card4 VARCHAR(25), card5 REAL, card6 VARCHAR(25),
    addr1 REAL, addr2 REAL,
    dist1 REAL, dist2 REAL,
    P_emaildomain VARCHAR(100),
    R_emaildomain VARCHAR(100)
);

-- 3. Create the raw staging table for Identity data
CREATE TABLE identities (
    TransactionID INT PRIMARY KEY,
    id_01 REAL, id_02 REAL, id_03 REAL, id_04 REAL, id_05 REAL, id_06 REAL, 
    id_07 REAL, id_08 REAL, id_09 REAL, id_10 REAL, id_11 REAL, 
    id_12 VARCHAR(20),
    DeviceType VARCHAR(50),
    DeviceInfo VARCHAR(255),
    CONSTRAINT fk_transaction FOREIGN KEY (TransactionID) REFERENCES transactions(TransactionID) ON DELETE CASCADE
);

-- 4. Create high-performance lookup indexes
CREATE INDEX idx_trans_dt ON transactions(TransactionDT);
CREATE INDEX idx_trans_card1 ON transactions(card1);
