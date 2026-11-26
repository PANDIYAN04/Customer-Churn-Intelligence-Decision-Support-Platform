
# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 2: DATASET UNDERSTANDING
# ==========================================

import pandas as pd
from pathlib import Path

# 1. Define dataset path
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED SUCCESSFULLY ==========\n")

# 3. Display first five records
print("FIRST FIVE RECORDS:")
print(df.head())

# 4. Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# 5. Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# 6. Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# 7. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 8. Check duplicate records
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# 9. Check duplicate customer IDs
print("\nDUPLICATE CUSTOMER IDs:")
print(df["customerID"].duplicated().sum())

# 10. Check churn distribution
print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGE:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 11. Check TotalCharges data quality
total_charges = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

print("\nINVALID TOTALCHARGES VALUES:")
print(total_charges.isnull().sum())

print("\n========== DATASET INSPECTION COMPLETED ==========")
