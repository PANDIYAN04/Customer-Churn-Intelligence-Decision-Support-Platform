# Customer Churn Intelligence & Decision Support Platform

**An End-to-End Machine Learning Solution for Customer Attrition Prediction, Risk Assessment, and Retention Analytics**

---

## 1. Project Overview

The **Customer Churn Intelligence & Decision Support Platform** is an end-to-end Machine Learning application designed to predict customer churn, evaluate customer risk levels, and support data-driven retention decisions.

Customer churn is a major challenge for businesses operating in subscription-based industries such as telecommunications, banking, insurance, and digital services. Identifying customers who are likely to discontinue a service allows organizations to take proactive measures to improve customer retention.

This project uses historical customer data to develop a predictive Machine Learning model that estimates the probability of customer churn. It combines predictive modeling, customer risk classification, exploratory data analysis, and interactive visualizations through a Streamlit dashboard.

Rather than providing only a churn prediction, the platform aims to transform customer data into meaningful business insights.

### LIVE PREVIEW LINK  : https://customer-churn-intelligence-decision-support-platform-ibm.streamlit.app/


## 2. Business Problem

Acquiring new customers is often more expensive than retaining existing ones.

However, businesses frequently struggle to identify customers who are likely to leave before they actually discontinue their services.

Traditional customer analysis methods may not effectively identify hidden patterns associated with customer attrition.

This project addresses the problem through Machine Learning by:

- Identifying customers who are more likely to churn.
- Estimating individual customer churn probabilities.
- Categorizing customers according to their churn risk.
- Analyzing factors associated with customer attrition.
- Supporting proactive customer retention strategies.

## 3. Project Objectives

The primary objectives are:

1. Perform data cleaning and preprocessing.
2. Conduct Exploratory Data Analysis (EDA).
3. Identify important customer characteristics associated with churn.
4. Develop and evaluate a Machine Learning classification model.
5. Generate cross-validation-based prediction probabilities.
6. Optimize the classification threshold using the F1-score.
7. Classify customers into different churn risk categories.
8. Develop an interactive Streamlit dashboard.
9. Present analytical insights through meaningful visualizations.
10. Support informed customer retention decisions.

## 4. Dataset Information

**Dataset:** Telco Customer Churn Dataset

The project uses a telecommunications customer dataset containing information about customer demographics, subscribed services, account details, billing information, and churn status.

| Attribute | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| SeniorCitizen | Senior citizen indicator |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| Tenure | Number of months with the company |
| PhoneService | Phone service subscription |
| InternetService | Internet service type |
| Contract | Customer contract type |
| MonthlyCharges | Monthly subscription charges |
| TotalCharges | Total accumulated charges |
| PaymentMethod | Customer payment method |
| Churn | Customer churn status (target variable) |

**Dataset dimensions:**

- Total records: 7,043
- Total features: 21
- Target variable: `Churn`
- Problem type: Binary Classification

**Target classes:**

- Yes – Customer has churned.
- No – Customer has not churned.

*Dataset source: IBM Telco Customer Churn sample dataset.*

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning and model evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Model serialization and loading |
| Git & GitHub | Version control and project hosting |

## 6. Machine Learning Workflow

The project follows a structured end-to-end Machine Learning workflow.

### Step 1: Data Collection

Load the telecommunications customer dataset and inspect its structure, dimensions, data types, and target distribution.

### Step 2: Data Preprocessing

- Handle missing and inconsistent values.
- Convert numerical and categorical variables into suitable formats.
- Encode categorical features.
- Prepare input features and target variables.
- Split the dataset into training and testing sets.

### Step 3: Exploratory Data Analysis

Analyze customer characteristics and identify patterns associated with churn.

The analysis includes:

- Customer churn distribution.
- Customer tenure analysis.
- Contract type analysis.
- Monthly charges distribution.
- Payment method analysis.
- Customer service subscription patterns.
- Relationships between customer characteristics and churn.

### Step 4: Model Development

Develop a Machine Learning classification model using the prepared training data.

The model learns patterns from historical customer information to estimate the likelihood of customer attrition.

### Step 5: Model Evaluation

Evaluate model performance using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (where implemented)

### Step 6: Cross-Validation and Threshold Optimization

Generate cross-validation prediction probabilities to identify a suitable classification threshold.

Instead of relying exclusively on the default threshold of 0.50, the project searches for a threshold that maximizes the F1-score.

**Selected classification threshold: 0.59**

This threshold is used to convert predicted probabilities into churn classifications.

### Step 7: Customer Risk Classification

Classify customers into three risk categories based on their predicted churn probabilities.

| Risk Category | Interpretation |
|---|---|
| Low Risk | Relatively low predicted churn probability |
| Medium Risk | Moderate predicted churn probability |
| High Risk | Relatively high predicted churn probability |

Risk boundaries should be configured in the application according to the implemented classification logic.

### Step 8: Decision Support

Use predicted churn probabilities and customer characteristics to support retention-related decisions.

Suggested actions include:

- Low Risk: Maintain regular customer engagement.
- Medium Risk: Monitor customer activity and improve engagement.
- High Risk: Prioritize customer outreach and investigate potential service concerns.

These recommendations are decision-support suggestions, not guaranteed retention outcomes.

## 7. Application Features

The Streamlit application provides an interactive interface for exploring customer churn patterns and model predictions.

### Dashboard and Analytics

- Overall customer statistics.
- Customer churn distribution.
- Churn-related analytical visualizations.
- Customer behavior analysis.
- Risk category distribution.
- Model performance insights.

### Data Visualizations

The application includes visual representations to improve understanding of customer behavior and churn patterns.

Examples include:

- Churn distribution charts.
- Customer tenure analysis.
- Monthly charges distribution.
- Contract-wise churn comparison.
- Customer risk distribution.
- Confusion matrix visualization.
- Feature importance visualization, if supported by the selected model.

### Predictive Analytics

- Customer churn probability estimation.
- Binary churn classification.
- Risk-level identification.
- Threshold-based prediction.
- Customer-level prediction insights.

## 8. Model Performance & Evaluation

![Model Performance & Evaluation Metrics]

The **Logistic Regression** classifier was evaluated on the held-out test set using an optimized decision threshold (`0.59`) to balance churn detection sensitivity with false-positive control:

<img width="1430" height="903" alt="Screenshot 2026-11-26 095312" src="https://github.com/user-attachments/assets/b5ad1ce3-f935-44d6-bbc0-aaa5596bbb28" />


| Metric | Score |
| :--- | :--- |
| **Accuracy** | `0.7324` |
| **Precision** | `0.4975` |
| **Recall** | `0.7914` |
| **F1-Score** | `0.6109` |
| **ROC-AUC** | `0.8421` |

* **Confusion Matrix (Optimized Threshold):** Correctly identified **271** churned customers (`TP`) and **815** retained customers (`TN`), with **103** missed churners (`FN`) and **220** false alarms (`FP`).
* **ROC Curve:** Achieves an AUC of **0.842**, demonstrating strong discriminative capability over the random classifier baseline[cite: 1, 2].

---

## Interactive Churn Prediction Interface

<img width="1791" height="938" alt="Screenshot 2026-11-26 095348" src="https://github.com/user-attachments/assets/3451febf-4949-4982-adf9-8b9597697725" />


![Customer Churn Prediction Input Interface]

The web application provides an interactive form to run real-time customer churn predictions based on the **IBM Telco Customer Churn** dataset:

* **Model Summary Sidebar:** Displays the active algorithm (`Logistic Regression`), decision threshold (`0.59`), test `ROC-AUC` (`0.8421`), and dataset context.
* **Customer Input Form:** Captures customer features across three categories:
  * **Personal Information:** Gender, senior citizen status, partner/dependents, and tenure (in months).
  * **Service Information:** Phone service, multiple lines, internet service type, online security, online backup, device protection, and tech support.
  * **Billing Information:** Contract type, paperless billing, payment method, monthly charges, and total charges[cite: 2].

**Optimized classification threshold:** 0.59

The classification threshold was selected through an F1-score optimization process using cross-validation prediction probabilities.

The following metrics should be reported using the actual final model evaluation results:

| Evaluation Metric | Result |
|---|---|
| Accuracy | Add actual result |
| Precision | Add actual result |
| Recall | Add actual result |
| F1-score | Add actual result |
| ROC-AUC | Add actual result, if available |


## 9. Project Architecture

The application follows this general workflow:

```text
Customer Dataset
       |
       v
Data Preprocessing
       |
       v
Exploratory Data Analysis
       |
       v
Feature Preparation
       |
       v
Machine Learning Model
       |
       v
Model Evaluation
       |
       v
Cross-Validation
       |
       v
Threshold Optimization
       |
       v
Churn Probability Prediction
       |
       v
Customer Risk Classification
       |
       v
Decision Support & Analytics
       |
       v
Streamlit Dashboard
```

## 10. Project Structure

The repository can be organized as follows:

```text
customer-churn-intelligence/
│
├── app.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── trained_model.pkl
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── images/
│   ├── dashboard.png
│   ├── churn_analysis.png
│   └── model_evaluation.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

*Adjust the folder names and filenames to match the actual repository. Include only files that exist in your project.*

## 11. Installation and Setup

Follow these instructions to run the application locally.

### Prerequisites

Install the following:

- Python 3.10 or later (compatible with the project dependencies).
- Visual Studio Code.
- Git.
- pip package manager.

### Step 1: Clone the Repository

```bash
git clone https://github.com/PANDIYAN04/customer-churn-intelligence.git
```

### Step 2: Navigate to the Project Directory

```bash
cd customer-churn-intelligence
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

### Step 6: Open the Dashboard

Open the following address in your browser:

```text
http://localhost:8501
```

The application should now be accessible locally.

## 12. Requirements

The project dependencies may include:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

Use the actual tested dependency versions in your final `requirements.txt`.

## 13. Business Applications

The proposed system can support several business use cases:

- Telecommunications customer retention.
- Subscription-based service analytics.
- Customer relationship management.
- Customer attrition monitoring.
- Customer risk segmentation.
- Data-driven retention planning.

The same general approach can be adapted to other industries by retraining the model with appropriate customer datasets.

## 14. Future Enhancements

Potential improvements include:

1. Deploy the application to Streamlit Community Cloud or another hosting platform.
2. Integrate real-time customer data.
3. Add customer-level explanations using SHAP.
4. Develop advanced customer segmentation.
5. Introduce personalized retention recommendations.
6. Add automated model monitoring.
7. Compare multiple Machine Learning algorithms.
8. Implement model retraining and performance tracking.
9. Introduce cost-sensitive churn prediction.
10. Develop a customer retention campaign analytics module.

## 15. Limitations

- Model predictions depend on the quality and representativeness of historical data.
- The dataset represents a particular telecommunications business context.
- Predicted churn probabilities are estimates, not guarantees.
- The optimized threshold may need adjustment for different business objectives.
- Retention recommendations require validation using actual business outcomes.
- Model performance may change when applied to different customer populations.

## 16. Author

**PANDIYAN M**

Aspiring Data Scientist | Machine Learning Enthusiast

**Areas of Interest:**
- Data Science
- Machine Learning
- Predictive Analytics
- Data Visualization
- Business Intelligence

**GitHub:** https://github.com/PANDIYAN04

**LinkedIn:** https://www.linkedin.com/in/pandiyan-m04

---

## 17. Project Summary

The Customer Churn Intelligence & Decision Support Platform demonstrates an end-to-end application of Machine Learning to a real-world business problem.

By combining customer data analysis, predictive modeling, cross-validation, threshold optimization, churn risk classification, and interactive visualization, the project illustrates how data science can support customer retention decisions.

**Core Concept:**

**Predict → Assess Risk → Analyze → Recommend**

---

*Developed as an applied Data Science and Machine Learning portfolio project.*
