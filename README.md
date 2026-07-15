# Telco Customer Churn Prediction — XGBoost vs CatBoost


An end-to-end customer churn project combining machine learning, model explainability, PySpark ETL, and Tableau business reporting.
(tableau/telco_churn_dashboard.png)

#  Project Overview
This repository develops and compares two classification pipelines for predicting customer churn:

- **XGBoost**, trained on encoded numerical features.
- **CatBoost**, trained directly on numerical and categorical features.

The project goes beyond model training by including threshold optimization, feature importance, SHAP explanations, prediction exports, a PySpark ETL demonstration, and a Tableau dashboard for communicating churn patterns and customer-risk segments.
## Business Problem

Customer churn reduces recurring revenue and increases acquisition costs. The project addresses three practical questions:

1. Which customers are most likely to leave?
2. Which customer characteristics are associated with higher churn?
3. How should the probability threshold be adjusted when the business prefers higher recall over fewer false-positive retention offers?

The model outputs can support targeted onboarding, contract-upgrade campaigns, technical-support interventions, and retention offers for customers with elevated churn probability.
## Dataset

The project uses the **Telco Customer Churn** dataset.

- **Customers:** 7,043
- **Original variables:** 21
- **Target:** `Churn`
- **Observed churn rate:** approximately 26.5%
- **Train/test split:** 80% / 20%, stratified by churn
- **Test customers:** 1,409

The variables include customer tenure, contract type, payment method, internet service, support services, monthly charges, total charges, and selected demographic attributes.
## Data Cleaning and Feature Preparation

The notebooks perform the following steps:

- Detect blank and missing values.
- Replace blank `TotalCharges` values for zero-tenure customers with `0`.
- Convert `TotalCharges` to a numerical variable.
- Validate selected numerical columns for invalid negative values.
- Convert the churn target to binary form.
- Apply binary and one-hot encoding for the XGBoost pipeline.
- Preserve native categorical variables for the CatBoost pipeline.
- Use a stratified train/test split to maintain the churn-class distribution.

For reporting and ETL, additional features include:

- `tenure_group`
- `high_monthly_charge`
- `churn_probability`
- `risk_segment`

Risk segments are defined from predicted churn probability:

| Risk segment | Churn probability |
|---|---:|
| Low Risk | 0.00–0.30 |
| Medium Risk | >0.30–0.70 |
| High Risk | >0.70 |

## Models

### XGBoost

XGBoost is trained using a randomized hyperparameter search with stratified 5-fold cross-validation.

Best configuration from the current notebook run:

```python
{
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
```

### CatBoost

CatBoost is used because it can process categorical variables directly without manually one-hot encoding every category.

Best configuration from the current notebook run:

```python
{
    "depth": 3,
    "learning_rate": 0.01,
    "iterations": 700,
    "l2_leaf_reg": 3
}
```
# Key Results (Test Set)
- Metric	XGBoost	CatBoost
- ROC-AUC	0.847	0.84
- Average Precision	0.658	0.668
- Accuracy @ 0.5	~0.82	0.80
- Accuracy (optimal)	~0.82 (F1/Youden)	0.789 (@ 0.36)
- F1 Score	~0.71	~0.70
- Precision (Churn)	~0.72	0.67 (0.5) / 0.58 (0.36)
- Recall (Churn)	~0.70	0.51 (0.5) / 0.72 (0.36)

 XGBoost: slightly higher AUC & Precision → fewer false positives.
 CatBoost: more flexible thresholding → better Recall when business goal is to catch more churners.
 ## Evaluation

Because churn is an imbalanced classification problem, the project evaluates more than accuracy. The main metrics are ROC-AUC, precision, recall, F1-score, confusion matrices, and precision-recall curves.

### Current test-set results at the default threshold of 0.50

| Model | ROC-AUC | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.847 | 0.799 | 0.66 | 0.50 | 0.57 |
| CatBoost | 0.845 | 0.804 | 0.67 | 0.52 | 0.59 |

### Results after Youden’s J threshold optimization

| Model | Threshold | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.318 | 0.778 | 0.560 | 0.757 | 0.644 |
| CatBoost | 0.296 | 0.762 | 0.536 | 0.778 | 0.635 |

The default threshold produces fewer false-positive churn alerts but misses many actual churners. Lowering the threshold substantially improves recall, which may be preferable when the cost of losing a customer is greater than the cost of making an unnecessary retention offer.
## Explainability

The project uses several methods to understand model behavior:

- XGBoost weight, gain, and cover importance.
- CatBoost `PredictionValuesChange` importance.
- Permutation importance.
- SHAP global summary plots.
- SHAP dependence plots for `tenure` and `MonthlyCharges`.
- Local SHAP explanations for individual customers.

The explainability analysis consistently highlights contract type and tenure as major churn drivers. Monthly charges, internet service, technical support, online security, total charges, and payment method also contribute to churn predictions.

## Business Insights

The analysis and dashboard indicate the following patterns:

- **Month-to-month contracts** have the highest churn rate.
- **Customers in their first 12 months** are the most vulnerable group.
- **Electronic-check users** show a higher churn rate than customers using automatic payment methods.
- **Long-tenure customers** are substantially more stable.
- Customers without **online security** or **technical support** tend to have higher churn risk.
- Higher monthly charges can increase risk, particularly when combined with short tenure and flexible contracts.

Recommended actions include improving early-stage onboarding, promoting longer contracts, encouraging automatic payments, bundling support services, and prioritizing customers in the high-risk segment.

## Databricks / PySpark ETL Demo

The `databrick.ipynb` notebook demonstrates a PySpark ETL workflow that can be adapted to Databricks.

It performs the following operations:

1. Reads the raw CSV dataset.
2. Removes duplicate rows.
3. Converts blank strings to null values.
4. Casts `TotalCharges` to a numerical type.
5. Fills selected missing numerical values.
6. Creates `tenure_group`.
7. Creates a `high_monthly_charge` flag using a threshold of 70.
8. Prepares the cleaned data for export.

> The uploaded notebook currently runs Spark locally on Windows. Its final CSV write cell failed because `HADOOP_HOME` was not configured. For a true Databricks workflow, replace the local Windows paths with DBFS, Unity Catalog, or workspace-volume paths.

## Tableau Dashboard

The Tableau dashboard is built from the scored test-set data and presents:

- **1,409 scored customers**
- **26.5% observed churn rate**
- **102 high-risk customers**
- Churn rate by contract type
- Churn rate by payment method
- Churn rate by tenure group
- Distribution of low-, medium-, and high-risk customers

The dashboard translates model outputs into a format that business users can use for retention prioritization.

### Dashboard data fields

The Tableau source should contain at least:

```text
customerID
Contract
PaymentMethod
tenure
tenure_group
MonthlyCharges
TotalCharges
actual_churn
predicted_churn
churn_probability
risk_segment
```
## Repository Structure

```text
customer-churn-prediction/
│
├── notebooks/
│   ├── 00_databricks_pyspark_etl_demo.ipynb
│   ├── 01_xgboost_churn.ipynb
│   └── 02_catboost_churn.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── xgboost_predictions.csv
│   ├── catboost_predictions.csv
│   ├── xgboost_shap_summary.png
│   └── catboost_shap_summary.png
│
├── tableau/
│   └── telco_churn_dashboard.png
│
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the raw CSV file in:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Use a relative path in the notebooks rather than a machine-specific Windows path.

### 5. Run the notebooks

```bash
jupyter notebook notebooks/01_xgboost_churn.ipynb
jupyter notebook notebooks/02_catboost_churn.ipynb
```

Run the PySpark demo separately:

```bash
jupyter notebook notebooks/00_databricks_pyspark_etl_demo.ipynb
```
### 6. Open the Tableau dashboard

Connect Tableau to the enriched CatBoost prediction file and refresh the dashboard after regenerating model predictions.

## Tools and Skills Demonstrated

- Python, pandas, NumPy
- scikit-learn
- XGBoost and CatBoost
- SHAP and permutation importance
- Hyperparameter tuning and stratified cross-validation
- Classification-threshold optimization
- PySpark ETL
- Tableau dashboard development
- Business-focused churn analysis

## Reproducibility Note

In the uploaded model notebooks, `RandomizedSearchCV` is currently fitted using the complete `X, y` dataset. This allows the held-out test observations to influence hyperparameter selection. Before treating the reported metrics as final, change the search calls to use only `X_train, y_train`, rerun both notebooks, and update the results table.

  #  Research
   - Rigorous methodology: stratified K-fold CV, AUC/AP focus under imbalance, threshold-free evaluation.
   - Interpretability with global + local SHAP, comparing intrinsic vs agnostic importance.
   - Foundation for further research: cost-sensitive learning, uplift modeling, FL (federated churn prediction).
   
  -  Author: Reyhaneh Ghaderi Chermahini
  -  Master’s in Data Science & Stochastic Processes   


