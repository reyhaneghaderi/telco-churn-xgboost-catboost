# Telco Customer Churn Prediction — XGBoost vs CatBoost
#  Project Overview
   - This repository presents two complete churn modeling pipelines on the Telco Customer Churn dataset:
   - XGBoost: industry-standard boosting model with strong performance and SHAP explainability.
   - CatBoost: state-of-the-art gradient boosting optimized for categorical features.
     
The goal is to evaluate, compare, and interpret both models for business decision-making and academic transparency.
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
