import pandas as pd
import numpy as np
import os

def make_dataset():
    # Define paths
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    
    # Load data
    print("Loading raw data...")
    claims = pd.read_csv(os.path.join(raw_dir, 'claims.csv'))
    policyholder = pd.read_csv(os.path.join(raw_dir, 'policyholder.csv'))
    third_party = pd.read_csv(os.path.join(raw_dir, 'third_party.csv'))
    
    # --- Step 1: Merging Strategy ---
    print("Merging data...")
    # Join Policy Data: LEFT JOIN on Policy_ID
    df = claims.merge(policyholder, on='Policy_ID', how='left', validate='many_to_one')
    
    # Verify every claim maps to a policy
    missing_policies = df['Customer_ID'].isna().sum()
    if missing_policies > 0:
        print(f"WARNING: {missing_policies} claims did not find a matching policy.")
    else:
        print("Verification: All claims successfully mapped to a policy.")
    
    # Join Third-Party Data: LEFT JOIN on Claim_ID
    df = df.merge(third_party, on='Claim_ID', how='left')
    
    # Fill missing values in third-party columns
    tp_cols = third_party.columns.difference(['Claim_ID'])
    for col in tp_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('None')
        else:
            df[col] = df[col].fillna(0)
            
    # --- Step 2: Cleaning & Filtering ---
    print("Cleaning and filtering...")
    # Target Check: Drop rows where Ultimate_Claim_Amount is null
    initial_len = len(df)
    df = df.dropna(subset=['Ultimate_Claim_Amount'])
    print(f"Dropped {initial_len - len(df)} rows with null Ultimate_Claim_Amount.")
    
    # Leakage Prevention: STRICTLY drop Settlement_Date and Estimated_Claim_Amount
    leaky_cols = ['Settlement_Date', 'Estimated_Claim_Amount']
    df = df.drop(columns=[col for col in leaky_cols if col in df.columns])
    
    # Imputation: Impute missing Age_of_Driver (Driver_Age) with median
    # Note: Column name in CSV is Age_of_Driver
    if 'Age_of_Driver' in df.columns:
        median_age = df['Age_of_Driver'].median()
        df['Age_of_Driver'] = df['Age_of_Driver'].fillna(median_age)
        print(f"Imputed missing Age_of_Driver with median: {median_age}")
    
    # --- Step 3: Feature Engineering ---
    print("Engineering features...")
    # Convert dates to datetime
    date_cols = ['FNOL_Date', 'Accident_Date', 'Policy_Start_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    # Days_to_Report: FNOL_Date - Accident_Date
    df['Days_to_Report'] = (df['FNOL_Date'] - df['Accident_Date']).dt.days
    
    # Policy_Tenure: If Policy_Start_Date exists
    if 'Policy_Start_Date' in df.columns:
        df['Policy_Tenure'] = (df['Accident_Date'] - df['Policy_Start_Date']).dt.days
    else:
        print("Note: Policy_Start_Date not found. Skipping Policy_Tenure calculation.")
        
    # TP_Severity_Score: mapping TP_Injury_Severity
    severity_map = {
        'Minor': 1,
        'Serious': 5,
        'Fatal': 10,
        'None': 0
    }
    # Note: Column name in CSV is TP_Injury_Severity
    if 'TP_Injury_Severity' in df.columns:
        df['TP_Severity_Score'] = df['TP_Injury_Severity'].map(severity_map).fillna(0)
        
    # Log_Target: log1p of Ultimate_Claim_Amount
    df['Log_Ultimate_Claim_Amount'] = np.log1p(df['Ultimate_Claim_Amount'])
    
    # Save processed data
    output_path = os.path.join(processed_dir, 'abt.csv')
    df.to_csv(output_path, index=False)
    print(f"ABT successfully saved to {output_path}")
    
    return df

if __name__ == '__main__':
    make_dataset()
