
# ============================================================
# CUSTOMER CHURN PREDICTION PROJECT
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "cleaned_customer_churn.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. VISUALIZATION SETTINGS
# ============================================================

sns.set_theme(style="whitegrid", palette="Set2")

plt.rcParams["figure.figsize"] = (10, 6)

plt.rcParams["font.size"] = 11


# ============================================================
# 3. LOAD CLEANED DATASET
# ============================================================

print("\n========== LOADING CLEANED DATASET ==========\n")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)

print("\nFirst five records:")

print(df.head())


# ============================================================
# 4. BASIC DATASET STATISTICS
# ============================================================

print("\n========== BASIC STATISTICS ==========\n")

total_customers = len(df)

churned_customers = (df["Churn"] == 1).sum()

retained_customers = (df["Churn"] == 0).sum()

churn_rate = (churned_customers / total_customers) * 100

retention_rate = (retained_customers / total_customers) * 100

print("Total customers:", total_customers)

print("Churned customers:", churned_customers)

print("Retained customers:", retained_customers)

print(f"Churn rate: {churn_rate:.2f}%")

print(f"Retention rate: {retention_rate:.2f}%")


# ============================================================
# 5. SAVE A HELPER FUNCTION FOR CHARTS
# ============================================================

def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", filename)


# ============================================================
# 6. CHART 1: CUSTOMER CHURN DISTRIBUTION
# ============================================================

print("\nGenerating Chart 1...")

churn_counts = df["Churn"].value_counts().sort_index()

labels = ["Retained", "Churned"]

plt.figure(figsize=(8, 6))

plt.pie(
    churn_counts.values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    explode=(0, 0.08)
)

plt.title("Customer Churn Distribution")

save_chart("01_churn_distribution.png")


# ============================================================
# 7. CHART 2: CHURN COUNT
# ============================================================

print("\nGenerating Chart 2...")

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    order=[0, 1],
    palette="Set2",
    legend=False
)

plt.xticks([0, 1], ["Retained", "Churned"])

plt.xlabel("Customer Status")

plt.ylabel("Number of Customers")

plt.title("Number of Retained and Churned Customers")

save_chart("02_churn_count.png")


# ============================================================
# 8. CHART 3: CHURN BY GENDER
# ============================================================

print("\nGenerating Chart 3...")

gender_churn = (
    df.groupby("gender")["Churn"]
    .mean() * 100
)

print("\nChurn percentage by gender:")

print(gender_churn)

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="gender",
    y="Churn",
    hue="gender",
    errorbar=None,
    legend=False
)

plt.ylabel("Average Churn (0 to 1)")

plt.title("Average Churn by Gender")

plt.xlabel("Gender")

plt.ylim(0, 1)

save_chart("03_churn_by_gender.png")


# ============================================================
# 9. CHART 4: CHURN BY SENIOR CITIZEN STATUS
# ============================================================

print("\nGenerating Chart 4...")

senior_churn = (
    df.groupby("SeniorCitizen")["Churn"]
    .mean() * 100
)

print("\nChurn percentage by senior citizen status:")

print(senior_churn)

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="SeniorCitizen",
    y="Churn",
    hue="SeniorCitizen",
    errorbar=None,
    legend=False
)

plt.xlabel("Senior Citizen")

plt.ylabel("Average Churn (0 to 1)")

plt.title("Churn by Senior Citizen Status")

plt.ylim(0, 1)

save_chart("04_churn_by_senior_citizen.png")


# ============================================================
# 10. CHART 5: CHURN BY CONTRACT TYPE
# ============================================================


# ============================================================
# 10. CHART 5: CHURN BY CONTRACT TYPE
# ============================================================

print("\nGenerating Chart 5...")

contract_churn = (
    df.groupby("Contract")["Churn"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nChurn percentage by contract:")
print(contract_churn)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=contract_churn.index,
    y=contract_churn.values
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 100)

save_chart("05_churn_by_contract.png")


# ============================================================
# 11. CHART 6: CHURN BY INTERNET SERVICE
# ============================================================

print("\nGenerating Chart 6...")


internet_churn = (
    df.groupby("InternetService")["Churn"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)


print("\nChurn percentage by internet service:")

print(internet_churn)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=internet_churn.index,
    y=internet_churn.values
)

plt.xlabel("Internet Service")

plt.ylabel("Churn Percentage (%)")

plt.title("Churn Rate by Internet Service")

save_chart("06_churn_by_internet_service.png")


# ============================================================
# 12. CHART 7: CHURN BY PAYMENT METHOD
# ============================================================

print("\nGenerating Chart 7...")

payment_churn = (
    df.groupby("PaymentMethod")["Churn"]
    .mean() 
    .mul(100)
    .sort_values(ascending=False)
)

print("\nChurn percentage by payment method:")

print(payment_churn)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=payment_churn.index,
    y=payment_churn.values
)

plt.xlabel("Payment Method")

plt.ylabel("Churn Percentage (%)")

plt.title("Churn Rate by Payment Method")

plt.xticks(rotation=20, ha="right")

save_chart("07_churn_by_payment_method.png")


# ============================================================
# 13. CHART 8: CUSTOMER TENURE DISTRIBUTION
# ============================================================

print("\nGenerating Chart 8...")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    multiple="stack"
)

plt.xlabel("Customer Tenure (Months)")

plt.ylabel("Number of Customers")

plt.title("Customer Tenure Distribution by Churn Status")

save_chart("08_tenure_distribution.png")


# ============================================================
# 14. CHART 9: MONTHLY CHARGES BY CHURN
# ============================================================

print("\nGenerating Chart 9...")

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    legend=False
)

plt.xticks([0, 1], ["Retained", "Churned"])

plt.xlabel("Customer Status")

plt.ylabel("Monthly Charges")

plt.title("Monthly Charges by Churn Status")

save_chart("09_monthly_charges_by_churn.png")


# ============================================================
# 15. CHART 10: TOTAL CHARGES BY CHURN
# ============================================================

print("\nGenerating Chart 10...")

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges",
    hue="Churn",
    legend=False
)

plt.xticks([0, 1], ["Retained", "Churned"])

plt.xlabel("Customer Status")

plt.ylabel("Total Charges")

plt.title("Total Charges by Churn Status")

save_chart("10_total_charges_by_churn.png")


# ============================================================
# 16. CHART 11: CUSTOMER SERVICE SUBSCRIPTIONS
# ============================================================

print("\nGenerating Chart 11...")

service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

service_churn = (
    df.groupby("Churn")[service_columns]
    .apply(
        lambda group: (
            group == "Yes"
        ).mean() * 100
    )
)

service_churn = service_churn.T

service_churn.columns = ["Retained", "Churned"]

print("\nService subscription percentages:")

print(service_churn)

service_churn.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.xlabel("Customer Services")

plt.ylabel("Customers with Service (%)")

plt.title("Service Subscription Comparison by Churn")

plt.xticks(rotation=35, ha="right")

plt.legend(title="Customer Status")

save_chart("11_service_subscription_comparison.png")


# ============================================================
# 17. CHART 12: NUMERICAL CORRELATION HEATMAP
# ============================================================

print("\nGenerating Chart 12...")

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

correlation_data = df[numerical_columns].copy()

correlation_data["SeniorCitizen"] = (
    correlation_data["SeniorCitizen"]
    .map({"No": 0, "Yes": 1})
)

correlation_matrix = correlation_data.corr()

print("\nCorrelation matrix:")

print(correlation_matrix)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Numerical Feature Correlation Heatmap")

save_chart("12_correlation_heatmap.png")


# ============================================================
# 18. ADDITIONAL BUSINESS INSIGHTS
# ============================================================

print("\n========== BUSINESS INSIGHTS ==========\n")

# Contract with highest churn

highest_contract = contract_churn.idxmax()

print(
    "Contract with highest churn:",
    highest_contract
)

# Internet service with highest churn

highest_internet = internet_churn.idxmax()

print(
    "Internet service with highest churn:",
    highest_internet
)

# Payment method with highest churn

highest_payment = payment_churn.idxmax()

print(
    "Payment method with highest churn:",
    highest_payment
)

# Average tenure by churn status

average_tenure = df.groupby("Churn")["tenure"].mean()

print("\nAverage tenure:")

print(average_tenure)

# Average monthly charges by churn status

average_monthly = (
    df.groupby("Churn")["MonthlyCharges"].mean()
)

print("\nAverage monthly charges:")

print(average_monthly)

# Average total charges by churn status

average_total = df.groupby("Churn")["TotalCharges"].mean()

print("\nAverage total charges:")

print(average_total)


# ============================================================
# 19. SAVE SUMMARY REPORT
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Churned Customers",
        "Retained Customers",
        "Churn Rate (%)",
        "Retention Rate (%)",
        "Average Tenure - Retained",
        "Average Tenure - Churned",
        "Average Monthly Charges - Retained",
        "Average Monthly Charges - Churned"
    ],

    "Value": [
        total_customers,
        churned_customers,
        retained_customers,
        round(churn_rate, 2),
        round(retention_rate, 2),
        round(average_tenure.loc[0], 2),
        round(average_tenure.loc[1], 2),
        round(average_monthly.loc[0], 2),
        round(average_monthly.loc[1], 2)
    ]
})

summary.to_csv(
    OUTPUT_DIR / "eda_summary.csv",
    index=False
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")

print("EXPLORATORY DATA ANALYSIS COMPLETED!")

print("Total visualizations generated: 12")

print("Output folder:", OUTPUT_DIR)

print("Summary report: eda_summary.csv")

print("==========================================\n")
