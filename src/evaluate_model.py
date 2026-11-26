
# ============================================================
# CUSTOMER CHURN PREDICTION - ADVANCED MODEL EVALUATION
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_churn_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_evaluation"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

print("\nLoading dataset and trained model...")

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

print("Dataset loaded:", df.shape)
print("Model loaded successfully!")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df = add_features(df)

X = df.drop(
    columns=["Churn", "customerID"]
)

y = df["Churn"].astype(int)


# ============================================================
# 4. RECREATE ORIGINAL TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 5. GET TEST PREDICTION PROBABILITIES
# ============================================================

print("\nGenerating test predictions...")

test_probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. GENERATE OUT-OF-FOLD TRAINING PREDICTIONS
# ============================================================

print("\nGenerating cross-validation probabilities...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Cross-validation predictions generated!")


# ============================================================
# 7. FIND THE BEST CLASSIFICATION THRESHOLD
# ============================================================

print("\nSearching for the best F1 threshold...")

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

threshold_results = []

for threshold in thresholds:

    train_predictions = (
        train_oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        train_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        train_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        train_predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["f1_score"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nBest threshold:", round(best_threshold, 2))
print("Training CV precision:", round(best_row["precision"], 4))
print("Training CV recall:", round(best_row["recall"], 4))
print("Training CV F1:", round(best_row["f1_score"], 4))


# ============================================================
# 8. COMPARE DEFAULT AND OPTIMIZED THRESHOLDS
# ============================================================

print("\nComparing thresholds...")

default_threshold = 0.50

default_predictions = (
    test_probabilities >= default_threshold
).astype(int)

optimized_predictions = (
    test_probabilities >= best_threshold
).astype(int)


def calculate_metrics(y_true, y_pred, probabilities):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        )
    }


default_metrics = calculate_metrics(
    y_test,
    default_predictions,
    test_probabilities
)

optimized_metrics = calculate_metrics(
    y_test,
    optimized_predictions,
    test_probabilities
)

comparison = pd.DataFrame(
    [
        {
            "threshold": default_threshold,
            **default_metrics
        },
        {
            "threshold": best_threshold,
            **optimized_metrics
        }
    ],
    index=[
        "Default threshold",
        "Optimized threshold"
    ]
)

print("\nTest performance comparison:")
print(comparison.round(4))


# ============================================================
# 9. CONFUSION MATRIX VISUALIZATIONS
# ============================================================

print("\nGenerating confusion matrix charts...")

for name, predictions in [
    ("default", default_predictions),
    ("optimized", optimized_predictions)
]:

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"]
    )

    plt.title(f"Confusion Matrix - {name.title()} Threshold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. ROC CURVE
# ============================================================

print("Generating ROC curve...")

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, test_probabilities):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. PRECISION-RECALL CURVE
# ============================================================

print("Generating Precision-Recall curve...")

pr_precision, pr_recall, pr_thresholds = (
    precision_recall_curve(
        y_test,
        test_probabilities
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    pr_recall,
    pr_precision
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. THRESHOLD ANALYSIS CHART
# ============================================================

print("Generating threshold analysis chart...")

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1_score"],
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 Across Thresholds")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. SAVE EVALUATION RESULTS
# ============================================================

print("\nSaving evaluation results...")

threshold_df.to_csv(
    OUTPUT_DIR / "threshold_analysis.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv"
)

# Save the chosen threshold for the future Streamlit app.

config = {
    "model_name": "Logistic Regression",
    "decision_threshold": best_threshold
}

with open(
    MODEL_DIR / "model_config.json",
    "w"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 EVALUATION COMPLETED!")
print("=" * 60)

print("Best training CV threshold:", round(best_threshold, 2))

print("\nDefault threshold test results:")
print(pd.Series(default_metrics).round(4))

print("\nOptimized threshold test results:")
print(pd.Series(optimized_metrics).round(4))

print("\nGenerated evaluation files:")
print("1. confusion_matrix_default.png")
print("2. confusion_matrix_optimized.png")
print("3. roc_curve.png")
print("4. precision_recall_curve.png")
print("5. threshold_analysis.png")
print("6. threshold_analysis.csv")
print("7. threshold_comparison.csv")
print("8. models/model_config.json")

print("\nEvaluation completed successfully!")
