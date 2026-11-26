
# =====================================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 3: DATA CLEANING AND PREPROCESSING
# =====================================================

import pandas as pd
from pathlib import Path

# -----------------------------------------------------
# 1. DEFINE PROJECT PATHS
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = PROCESSED_DIR / "cleaned_customer_churn.csv"

# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# -----------------------------------------------------

print("\n========== LOADING RAW DATA ==========\n")

df = pd.read_csv(RAW_PATH)

print("Original dataset shape:", df.shape)


# -----------------------------------------------------
# 3. STANDARDIZE COLUMN NAMES AND TEXT VALUES
# -----------------------------------------------------

print("\n========== STANDARDIZING TEXT ==========\n")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Remove unnecessary spaces from string values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

print("Text standardization completed.")


# -----------------------------------------------------
# 4. INSPECT MISSING TOTALCHARGES VALUES
# -----------------------------------------------------

print("\n========== INVESTIGATING MISSING VALUES ==========\n")

# Convert TotalCharges to numeric
# Invalid or blank entries become NaN

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

missing_count = df["TotalCharges"].isnull().sum()

print("Missing TotalCharges:", missing_count)

# Check tenure for customers with missing TotalCharges

missing_customers = df[df["TotalCharges"].isnull()]

print("\nCustomers with missing TotalCharges:")

print(
    missing_customers[
        ["customerID", "tenure", "TotalCharges"]
    ]
)

# Verify whether all missing values belong to
# customers with zero tenure

if missing_count > 0:

    if (missing_customers["tenure"] == 0).all():

        print(
            "\nAll missing TotalCharges values "
            "belong to customers with zero tenure."
        )

        # Replace missing TotalCharges with zero
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

        print("Missing TotalCharges replaced with 0.")

    else:

        raise ValueError(
            "Unexpected missing TotalCharges values. "
            "Investigate before imputing."
        )

else:

    print("No missing TotalCharges values found.")


# -----------------------------------------------------
# 5. CHECK AND REMOVE DUPLICATE ROWS
# -----------------------------------------------------

print("\n========== CHECKING DUPLICATES ==========\n")

duplicate_count = df.duplicated().sum()

print("Duplicate rows found:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")


# -----------------------------------------------------
# 6. CHECK CUSTOMER ID DUPLICATES
# -----------------------------------------------------

print("\n========== CHECKING CUSTOMER IDS ==========\n")

duplicate_ids = df["customerID"].duplicated().sum()

print("Duplicate customer IDs:", duplicate_ids)

if duplicate_ids > 0:

    raise ValueError(
        "Duplicate customer IDs detected. "
        "Investigate before continuing."
    )

else:

    print("All customer IDs are unique.")


# -----------------------------------------------------
# 7. CONVERT SENIOR CITIZEN INTO CATEGORICAL VALUES
# -----------------------------------------------------

print("\n========== CONVERTING CATEGORICAL DATA ==========\n")

df["SeniorCitizen"] = df["SeniorCitizen"].map({
    0: "No",
    1: "Yes"
})

if df["SeniorCitizen"].isnull().any():

    raise ValueError(
        "Unexpected SeniorCitizen values detected."
    )

print("SeniorCitizen converted into Yes/No.")


# -----------------------------------------------------
# 8. CONVERT TARGET VARIABLE INTO NUMERICAL FORMAT
# -----------------------------------------------------

print("\n========== ENCODING TARGET VARIABLE ==========\n")

# No = 0
# Yes = 1

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if df["Churn"].isnull().any():

    raise ValueError(
        "Unexpected Churn values detected."
    )

df["Churn"] = df["Churn"].astype("int64")

print("Churn target encoding completed.")

print("\nChurn distribution:")

print(df["Churn"].value_counts())


# -----------------------------------------------------
# 9. VERIFY DATA TYPES
# -----------------------------------------------------

print("\n========== DATA TYPES ==========\n")

print(df.dtypes)


# -----------------------------------------------------
# 10. VERIFY MISSING VALUES
# -----------------------------------------------------

print("\n========== FINAL MISSING VALUE CHECK ==========\n")

print(df.isnull().sum())

total_missing = df.isnull().sum().sum()

print("\nTotal missing values:", total_missing)

if total_missing > 0:

    raise ValueError(
        "Missing values remain in the cleaned dataset."
    )


# -----------------------------------------------------
# 11. FINAL DATASET VALIDATION
# -----------------------------------------------------

print("\n========== FINAL VALIDATION ==========\n")

print("Final dataset shape:", df.shape)

print("Duplicate rows:", df.duplicated().sum())

print("Unique customers:", df["customerID"].nunique())

print("Churn values:", sorted(df["Churn"].unique()))

print("\nFirst five cleaned records:")

print(df.head())


# -----------------------------------------------------
# 12. SAVE CLEANED DATASET
# -----------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n========== DATA CLEANING COMPLETED ==========\n")

print("Cleaned dataset saved successfully!")

print("Output location:", OUTPUT_PATH)
