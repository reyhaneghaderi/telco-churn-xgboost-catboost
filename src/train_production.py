from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "churn_xgb_bundle.joblib"
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Clean TotalCharges
# --------------------------------------------------

df.loc[
    df["tenure"] == 0,
    "TotalCharges"
] = 0

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"]
)


# --------------------------------------------------
# 4. Create X and y
# --------------------------------------------------

X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)


# --------------------------------------------------
# 5. Split train+validation / test
# --------------------------------------------------

X_train_val, X_test, y_train_val, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
)


# --------------------------------------------------
# 6. Split train / validation
# --------------------------------------------------

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        stratify=y_train_val,
        random_state=42
    )
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_val)
)

print(
    "Test rows:",
    len(X_test)
)


# --------------------------------------------------
# 7. Detect categorical and numeric columns
# --------------------------------------------------

categorical_cols = (
    X_train
    .select_dtypes(include="object")
    .columns
    .tolist()
)

numeric_cols = (
    X_train
    .select_dtypes(exclude="object")
    .columns
    .tolist()
)

print("\nCategorical columns:")
print(categorical_cols)

print("\nNumeric columns:")
print(numeric_cols)


# --------------------------------------------------
# 8. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols,
        ),

        (
            "numeric",
            "passthrough",
            numeric_cols,
        ),
    ]
)


# --------------------------------------------------
# 9. XGBoost model
# --------------------------------------------------

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,

    # Use one worker
    n_jobs=1,
)


# --------------------------------------------------
# 10. Combine preprocessing + XGBoost
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb
        ),
    ]
)


# --------------------------------------------------
# 11. Hyperparameter search
# --------------------------------------------------

param_grid = {

    "model__max_depth":
        [3, 5, 7],

    "model__learning_rate":
        [0.1, 0.01, 0.001],

    "model__n_estimators":
        [50, 100, 200],

    "model__subsample":
        [0.8, 1.0],

    "model__colsample_bytree":
        [0.8, 1.0],
}


# Use 3-fold cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    # Test 10 random combinations
    n_iter=10,

    scoring="roc_auc",

    cv=cv,

    # IMPORTANT:
    # do not use -1 on your laptop
    n_jobs=1,

    refit=True,

    random_state=42,

    verbose=2,
)


# Search ONLY on training data
search.fit(
    X_train,
    y_train
)


# Best model
best_model = search.best_estimator_


print(
    "\nBest CV ROC-AUC:"
)

print(
    search.best_score_
)


print(
    "\nBest parameters:"
)

print(
    search.best_params_
)


# --------------------------------------------------
# 12. Choose threshold using validation data
# --------------------------------------------------

val_probability = (
    best_model
    .predict_proba(X_val)[:, 1]
)


fpr, tpr, thresholds = roc_curve(
    y_val,
    val_probability
)


youden_j = tpr - fpr


best_threshold = float(
    thresholds[
        np.argmax(youden_j)
    ]
)


print(
    "\nSelected threshold:"
)

print(
    best_threshold
)


# --------------------------------------------------
# 13. Evaluate on TEST data
# --------------------------------------------------

test_probability = (
    best_model
    .predict_proba(X_test)[:, 1]
)


test_prediction = (
    test_probability
    >= best_threshold
).astype(int)


roc_auc = roc_auc_score(
    y_test,
    test_probability
)


accuracy = accuracy_score(
    y_test,
    test_prediction
)


precision = precision_score(
    y_test,
    test_prediction
)


recall = recall_score(
    y_test,
    test_prediction
)


f1 = f1_score(
    y_test,
    test_prediction
)


print(
    "\n========== TEST RESULTS =========="
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1:",
    round(f1, 4)
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        test_prediction
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        test_prediction,
        digits=3
    )
)


# --------------------------------------------------
# 14. Save model for FastAPI
# --------------------------------------------------

bundle = {

    "pipeline":
        best_model,

    "threshold":
        best_threshold,

    "input_columns":
        X.columns.tolist(),
}


joblib.dump(
    bundle,
    MODEL_PATH
)


print(
    "\nModel saved successfully:"
)

print(
    MODEL_PATH
)