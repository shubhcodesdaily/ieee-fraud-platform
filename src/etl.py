import os #Used to check the existence of the file
import pandas as pd
import psycopg2 #This will help to act as a translator between python and PostgreSQL
from psycopg2.extras import execute_values #help to insert multiple rows at once


os.chdir(r"C:\Users\SHUBH KESHRI\Documents\IEEE-CIS Fraud Detection\ieee-fraud-platform")

# 2. Check if 'data' folder exists
if os.path.exists("data"):
    print("Data folder found! Inside it, there are these files:")
    print(os.listdir("data"))
else:
    print("Cannot even find the 'data' folder from here.")

# Database address and credentials
DB_HOST = "localhost"
DB_NAME = "sentinel_db"
DB_USER = "postgres"
DB_PASSWORD = "Atomic@123"

#Ingestion of data from CSV file to PostgreSQL database
def ingestion(file_path, table_name, columns_to_keep):
    #Check if the file exists for ingestion and return a message if it does not exist
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    else:
        print(f"File {file_path} exists. Proceeding with ingestion.")

    #Connect to the PostgreSQL database
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cursor = conn.cursor()
    for chunk in pd.read_csv(file_path, chunksize=50000, usecols=columns_to_keep):
        chunk = chunk.where(pd.notnull(chunk), None)
        columns = list(chunk.columns)
        values = [tuple(row) for row in chunk.to_numpy()]
        query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES %s ON CONFLICT (TransactionID) DO NOTHING"
        execute_values(cursor, query, values)
        conn.commit()
    cursor.close()
    conn.close()
    
    
if __name__ == "__main__":
    # Define columns to extract
    tx_cols = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 
               'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 
               'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain']
    
    id_cols = ['TransactionID', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 
               'id_12', 'DeviceType', 'DeviceInfo']

    # Call your ingestion function using standard CSV paths
    ingestion("data/train_transaction.csv", "transactions", tx_cols)
    ingestion("data/train_identity.csv", "identities", id_cols)    

        
    
        