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

 # What’s Inside

- Clean, reproducible notebooks (xgboost-churn.ipynb, catboost-churn.ipynb)
- Stratified K-Fold CV with hyperparameter tuning (RandomizedSearchCV)
- Model evaluation with ROC, PR, confusion matrices at multiple thresholds
- Feature importance:

- XGBoost Gain vs Permutation Importance

- CatBoost PredictionValuesChange vs Permutation Importance
   - SHAP explainability (global & local): summary, dependence, force/waterfall plots
   - Business reporting: actionable retention strategies
   - Visualizations: calibration plots, cumulative gains, threshold optimization

 # Feature Importance Comparison
Rank	XGBoost (Gain)	CatBoost (PredictionValuesChange)
 - Contract (Month-to-month)	Contract
 - Tenure	Tenure
 - OnlineSecurity = No	InternetService
 - InternetService = Fiber optic	TechSupport
 - MonthlyCharges	MonthlyCharges
 - PaymentMethod = E-check	OnlineSecurity
 - TotalCharges	TotalCharges
 - TechSupport = No	PaymentMethod

 # Insights :
  - Contract type & tenure dominate churn risk.
  - MonthlyCharges > €95 sharply increases churn probability (especially month-to-month).
  - Fiber optic internet and lack of security/tech support raise churn risk.
  - Demographics (gender, partner, senior status) contribute little.
 # Business Insights
  - New customers (<12 months, month-to-month): highest risk → target with onboarding & discounts.
  - Tenure 12–24 months: churn transition zone → proactive retention offers.
  - Loyal customers (>60 months): stable → maintain satisfaction, avoid price shocks.

     Threshold choice:

  - 0.5 → precision focus (fewer wasted offers).
  - 0.36 → recall focus (catch more churners).
  - PR curve shows top 10–20% high-risk customers can be targeted with ~80–90% precision.
  # Visuals Included (images/)

   - ROC & Precision–Recall curves
   - SHAP summary & dependence plots (MonthlyCharges, tenure × Contract)
   - Feature importance (XGBoost vs CatBoost vs Permutation)
   - Confusion matrices (0.5 vs optimized thresholds)
   - Calibration & cumulative gains charts

  # For Academia (PhD / Research)
   - Rigorous methodology: stratified K-fold CV, AUC/AP focus under imbalance, threshold-free evaluation.
   - Interpretability with global + local SHAP, comparing intrinsic vs agnostic importance.
   - Foundation for further research: cost-sensitive learning, uplift modeling, FL (federated churn prediction).
     
   # Skills & Tools Demonstrated
     - ML frameworks: XGBoost, CatBoost, scikit-learn
     - Model explainability: SHAP, permutation importance
     - Business analytics: threshold tuning, lift/gain analysis, ROI-driven strategies
     - Python stack: pandas, matplotlib, seaborn, shap, catboost, xgboost
     - Communication: turning technical findings into retention actions  

   #  How to Run
      - pip install -r requirements.txt
      - jupyter notebook notebooks/xgboost-churn.ipynb
      -jupyter notebook notebooks/catboost-churn.ipynb


  -  Author: Reyhaneh Ghaderi Chermahini
  -  Master’s in Data Science & Stochastic Processes   


