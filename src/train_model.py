
# ============================================================
# CUSTOMER CHURN PREDICTION - MODEL TRAINING
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.features import add_features


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_customer_churn.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading cleaned customer churn dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nApplying feature engineering...")

df = add_features(df)

print("Feature engineering completed!")

print("\nNew feature created: tenure_group")

print(df[["tenure", "tenure_group"]].head())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

print("\nSeparating features and target...")

# Remove customerID because it is only an identifier.
# Churn is the target variable, not an input feature.

X = df.drop(
    columns=["Churn", "customerID"]
)

y = df["Churn"].astype(int)

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nChurn percentages:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting dataset into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting churn distribution:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

print("\nIdentifying numerical and categorical columns...")

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. BUILD PREPROCESSING PIPELINES
# ============================================================

print("\nBuilding preprocessing pipelines...")

# Numerical preprocessing:
# 1. Fill missing values using median.
# 2. Standardize numerical values.

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing:
# 1. Fill missing values using the most frequent value.
# 2. Convert categories into numerical columns.

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Combine both preprocessing pipelines.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)

print("Preprocessing pipeline created successfully!")


# ============================================================
# 8. DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\nDefining machine learning models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 9. TRAINING AND CROSS-VALIDATION
# ============================================================

print("\nStarting model training and cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = {}

trained_pipelines = {}

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Evaluating:", model_name)
    print("=" * 55)

    # Combine preprocessing and model into one pipeline.

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Cross-validation is performed only on training data.
    # Each fold learns its own preprocessing transformations.

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    mean_f1 = scores.mean()
    std_f1 = scores.std()

    cv_results[model_name] = {
        "mean_f1": mean_f1,
        "std_f1": std_f1
    }

    trained_pipelines[model_name] = pipeline

    print("F1 scores:", scores)
    print(f"Mean F1 score: {mean_f1:.4f}")
    print(f"Standard deviation: {std_f1:.4f}")


# ============================================================
# 10. SELECT THE BEST MODEL
# ============================================================

print("\nComparing cross-validation results...")

cv_table = pd.DataFrame(cv_results).T

cv_table = cv_table.sort_values(
    by="mean_f1",
    ascending=False
)

print("\nModel comparison:")
print(cv_table)

best_model_name = cv_table.index[0]

best_pipeline = trained_pipelines[best_model_name]

print("\nSelected model:", best_model_name)


# ============================================================
# 11. TRAIN SELECTED MODEL ON FULL TRAINING DATA
# ============================================================

print("\nTraining selected model on full training dataset...")

best_pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 12. EVALUATE MODEL ON TEST DATA
# ============================================================

print("\nEvaluating model on unseen test data...")

y_pred = best_pipeline.predict(X_test)

y_probability = best_pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 13. PRINT EVALUATION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print("Selected model:", best_model_name)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nDetailed Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Retained", "Churned"],
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 15. SAVE MODEL AND RESULTS
# ============================================================

print("\nSaving trained model...")

model_path = MODEL_DIR / "best_churn_model.joblib"

joblib.dump(
    best_pipeline,
    model_path
)

print("Model saved at:", model_path)


# Save cross-validation comparison.

cv_table.to_csv(
    OUTPUT_DIR / "model_comparison.csv"
)


# Save evaluation metrics.

evaluation_results = {
    "selected_model": best_model_name,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

pd.DataFrame(
    [evaluation_results]
).to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)


# Save confusion matrix.

pd.DataFrame(
    cm,
    index=["Actual Retained", "Actual Churned"],
    columns=["Predicted Retained", "Predicted Churned"]
).to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


# Save detailed classification report.

report = classification_report(
    y_test,
    y_pred,
    target_names=["Retained", "Churned"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nBest model:", best_model_name)
print(f"Test F1 Score: {f1:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

print("\nGenerated files:")
print("1. models/best_churn_model.joblib")
print("2. outputs/model_evaluation/model_comparison.csv")
print("3. outputs/model_evaluation/evaluation_metrics.csv")
print("4. outputs/model_evaluation/confusion_matrix.csv")
print("5. outputs/model_evaluation/classification_report.csv")

print("\nMachine learning training completed!")
