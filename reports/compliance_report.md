# Model Explainability & Compliance Report

## 1. Compliance Check
[PASS] No prohibited variables (Race, Religion, etc.) were found in the model features.

## 2. Top 5 Cost Drivers (SHAP Analysis)
The following features have the most significant impact on predicting claim costs:

- **Claim_Type_Fire**: Average impact (SHAP value) of 1.3764
- **Severity_Band_Minor**: Average impact (SHAP value) of 0.6730
- **Claim_Type_Theft**: Average impact (SHAP value) of 0.4266
- **Claim_Type_Vandalism**: Average impact (SHAP value) of 0.1792
- **Severity_Band_Moderate**: Average impact (SHAP value) of 0.1707

## 3. Summary Visualization
The SHAP summary plot visualizes how feature values influence the prediction (driving costs up or down). You can find the plot in: `reports/figures/shap_summary_plot.png`.
