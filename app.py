
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.features import add_features


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_customer_churn.csv"
)

EVALUATION_DIR = BASE_DIR / "outputs" / "model_evaluation"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return model, config


@st.cache_data
def load_customer_data():
    return pd.read_csv(DATA_PATH)


try:
    model, model_config = load_model()
    historical_data = load_customer_data()

    THRESHOLD = float(
        model_config["decision_threshold"]
    )

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load the project files: {error}")
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_risk_category(probability):

    if probability >= 0.70:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    return "Low"


def show_customer_visualizations(
    probability,
    customer_data,
    historical_data
):

    st.markdown("---")
    st.header("📊 Customer Analytics")

    churn_probability = probability * 100
    retention_probability = (1 - probability) * 100

    # --------------------------------------------------------
    # CHART 1: CHURN PROBABILITY GAUGE
    # --------------------------------------------------------

    st.subheader("1. Churn Probability vs Decision Threshold")

    fig, ax = plt.subplots(figsize=(10, 2.5))

    ax.barh(
        ["Customer"],
        [1],
        color="#E5E7EB"
    )

    ax.barh(
        ["Customer"],
        [probability],
        color="#E76F51"
    )

    ax.axvline(
        THRESHOLD,
        color="#2563EB",
        linestyle="--",
        linewidth=2,
        label=f"Decision threshold: {THRESHOLD:.0%}"
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Churn Probability")
    ax.set_title("Customer Churn Risk")
    ax.legend(loc="lower right")

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"The customer has a {churn_probability:.2f}% "
        f"estimated churn probability. The model's decision "
        f"threshold is {THRESHOLD:.0%}."
    )

    # --------------------------------------------------------
    # CHART 2: CHURN VS RETENTION DONUT
    # --------------------------------------------------------

    st.subheader("2. Predicted Churn vs Retention")

    col1, col2 = st.columns([1, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.pie(
            [churn_probability, retention_probability],
            labels=["Churn", "Retention"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "width": 0.40,
                "edgecolor": "white"
            }
        )

        ax.set_title("Prediction Probability Distribution")

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

        st.metric(
            "Retention Probability",
            f"{retention_probability:.2f}%"
        )

        st.info(
            "These values represent the model's estimated "
            "probabilities, not guaranteed customer outcomes."
        )

    # --------------------------------------------------------
    # CHART 3: CUSTOMER TENURE COMPARISON
    # --------------------------------------------------------

    st.subheader("3. Customer Tenure Comparison")

    retained_tenure = historical_data.loc[
        historical_data["Churn"] == 0,
        "tenure"
    ]

    churned_tenure = historical_data.loc[
        historical_data["Churn"] == 1,
        "tenure"
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(
        [retained_tenure, churned_tenure],
        labels=["Retained", "Churned"],
        patch_artist=True
    )

    customer_tenure = customer_data["tenure"]

    ax.scatter(
        [1, 2],
        [customer_tenure, customer_tenure],
        color="red",
        marker="D",
        s=90,
        label="Selected Customer",
        zorder=5
    )

    ax.set_ylabel("Tenure (Months)")
    ax.set_title("Tenure Distribution by Historical Churn Group")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The box plots show historical customer tenure "
        "distributions. Red diamonds indicate the selected "
        "customer's tenure."
    )

    # --------------------------------------------------------
    # CHART 4: MONTHLY CHARGES COMPARISON
    # --------------------------------------------------------

    st.subheader("4. Monthly Charges Comparison")

    retained_charges = historical_data.loc[
        historical_data["Churn"] == 0,
        "MonthlyCharges"
    ]

    churned_charges = historical_data.loc[
        historical_data["Churn"] == 1,
        "MonthlyCharges"
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(
        [retained_charges, churned_charges],
        labels=["Retained", "Churned"],
        patch_artist=True
    )

    customer_charges = customer_data["MonthlyCharges"]

    ax.scatter(
        [1, 2],
        [customer_charges, customer_charges],
        color="red",
        marker="D",
        s=90,
        label="Selected Customer",
        zorder=5
    )

    ax.set_ylabel("Monthly Charges")
    ax.set_title("Monthly Charges by Historical Churn Group")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Compare the selected customer's monthly charges "
        "with the historical retained and churned groups."
    )

    # --------------------------------------------------------
    # CHART 5: CONTRACT DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("5. Historical Churn by Contract Type")

    contract_churn = (
        historical_data.groupby("Contract", observed=False)["Churn"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        contract_churn.index,
        contract_churn.values
    )

    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    ax.set_title("Historical Churn Rate by Contract Type")

    for index, value in enumerate(contract_churn.values):

        ax.text(
            index,
            value + 1,
            f"{value:.1f}%",
            ha="center"
        )

    ax.set_ylim(0, max(contract_churn.values) + 12)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This chart shows historical churn rates for each "
        "contract type in the dataset."
    )


# ============================================================
# 5. MODEL PERFORMANCE SECTION
# ============================================================

def show_model_performance():

    st.markdown("---")
    st.header("🎯 Machine Learning Model Performance")

    st.write(
        "The following results come from the model evaluation "
        "performed on the held-out test dataset."
    )

    metric_file = EVALUATION_DIR / "evaluation_metrics.csv"

    if metric_file.exists():

        metrics = pd.read_csv(metric_file)

        st.subheader("Evaluation Metrics")

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

    # Display evaluation plots if available

    chart_files = [
        (
            "confusion_matrix_optimized.png",
            "Confusion Matrix"
        ),
        (
            "roc_curve.png",
            "ROC Curve"
        ),
        (
            "precision_recall_curve.png",
            "Precision-Recall Curve"
        ),
        (
            "threshold_analysis.png",
            "Decision Threshold Analysis"
        )
    ]

    for index in range(0, len(chart_files), 2):

        col1, col2 = st.columns(2)

        for col, item in zip(
            [col1, col2],
            chart_files[index:index + 2]
        ):

            filename, title = item
            image_path = EVALUATION_DIR / filename

            with col:

                if image_path.exists():

                    st.subheader(title)

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )


# ============================================================
# 6. MAIN DASHBOARD HEADER
# ============================================================

st.title("📊 AI-Powered Customer Churn Prediction System")

st.markdown(
    """
    This application uses Machine Learning to estimate the
    probability of customer churn.

    Enter customer information below to generate a prediction
    and explore relevant historical customer analytics.
    """
)

st.markdown("---")


# ============================================================
# 7. MODEL INFORMATION
# ============================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.write(
    f"**Model:** {model_config.get('model_name', 'Unknown')}"
)

st.sidebar.write(
    f"**Decision Threshold:** {THRESHOLD:.2f}"
)

st.sidebar.write(
    "**Test ROC-AUC:** 0.8421"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This model uses the IBM Telco Customer Churn dataset."
)

st.sidebar.caption(
    "Predictions are estimates and should support, "
    "not replace, business decisions."
)


# ============================================================
# 8. CUSTOMER INPUT FORM
# ============================================================

st.header("👤 Enter Customer Information")

with st.form("customer_prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    with col1:

        st.subheader("Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

    # --------------------------------------------------------
    # SERVICE INFORMATION
    # --------------------------------------------------------

    with col2:

        st.subheader("Service Information")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines_options = (
            ["No phone service", "No", "Yes"]
            if phone_service == "No"
            else ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            multiple_lines_options
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        if internet_service == "No":

            internet_options = ["No internet service"]

        else:

            internet_options = ["Yes", "No"]

        online_security = st.selectbox(
            "Online Security",
            internet_options
        )

        online_backup = st.selectbox(
            "Online Backup",
            internet_options
        )

        device_protection = st.selectbox(
            "Device Protection",
            internet_options
        )

        tech_support = st.selectbox(
            "Tech Support",
            internet_options
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            internet_options
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            internet_options
        )

    # --------------------------------------------------------
    # BILLING INFORMATION
    # --------------------------------------------------------

    with col3:

        st.subheader("Billing Information")

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# ============================================================
# 9. PREDICTION AND RESULTS
# ============================================================

if submitted:

    # Validate customer information

    if tenure == 0 and total_charges != 0:

        st.error(
            "For a customer with 0 months of tenure, "
            "Total Charges should be 0."
        )

        st.stop()

    if tenure > 0 and monthly_charges > 0 and total_charges == 0:

        st.error(
            "Total Charges cannot be 0 when tenure and "
            "Monthly Charges are positive."
        )

        st.stop()

    # Build customer record

    customer_data = {

        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges

    }

    # Convert dictionary to DataFrame

    input_df = pd.DataFrame([customer_data])

    # Feature engineering

    input_df = add_features(input_df)

    # Predict churn probability

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Apply optimized decision threshold

    prediction = int(
        probability >= THRESHOLD
    )

    risk_category = get_risk_category(probability)

    # --------------------------------------------------------
    # DISPLAY PREDICTION RESULTS
    # --------------------------------------------------------

    st.markdown("---")

    st.header("📈 Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with result_col2:

        st.metric(
            "Risk Category",
            risk_category
        )

    with result_col3:

        prediction_text = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        st.metric(
            "Prediction",
            prediction_text
        )

    st.progress(
        probability,
        text="Estimated probability of customer churn"
    )

    # --------------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown("---")

    st.header("💡 Business Recommendation")

    if prediction == 1:

        st.warning(
            "This customer is classified as likely to churn "
            "at the selected decision threshold."
        )

        st.write("**Suggested actions:**")

        st.markdown(
            """
            - Contact the customer to understand their concerns.
            - Review available contract renewal offers.
            - Investigate whether service quality is an issue.
            - Consider an appropriate customer retention offer.
            """
        )

    else:

        st.success(
            "This customer is classified as likely to stay "
            "at the selected decision threshold."
        )

        st.write("**Suggested actions:**")

        st.markdown(
            """
            - Continue providing consistent customer support.
            - Monitor customer satisfaction.
            - Offer relevant upgrades when appropriate.
            - Maintain engagement through loyalty programs.
            """
        )

    # --------------------------------------------------------
    # CUSTOMER VISUALIZATIONS
    # --------------------------------------------------------

    show_customer_visualizations(
        probability,
        customer_data,
        historical_data
    )

    # --------------------------------------------------------
    # SUBMITTED CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    with st.expander("View Submitted Customer Information"):

        st.dataframe(
            pd.DataFrame([customer_data]),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 10. MODEL PERFORMANCE DASHBOARD
# ============================================================

show_model_performance()


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Powered Customer Churn Prediction & Business "
    "Intelligence System | Built with Python, Scikit-learn "
    "and Streamlit"
)
