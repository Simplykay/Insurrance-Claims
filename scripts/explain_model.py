import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

def generate_explainability_report():
    # Define paths
    processed_data_path = 'data/processed/abt.csv'
    model_path = 'src/models/model.pkl'
    figures_dir = 'reports/figures'
    report_path = 'reports/compliance_report.md'
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Load model
    print("Loading model...")
    pipeline = joblib.load(model_path)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(processed_data_path)
    
    # Identify protected attributes (Compliance Check)
    prohibited_vars = ['Race', 'Religion', 'Ethnicity', 'Political_Affiliation']
    found_prohibited = [v for v in prohibited_vars if v in df.columns]
    
    # Define features used in training (consistent with train_model.py)
    drop_cols = [
        'Claim_ID', 'Policy_ID', 'Customer_ID', 'Accident_Date', 'FNOL_Date',
        'Ultimate_Claim_Amount', 'Log_Ultimate_Claim_Amount', 'Status',
        'TP_ID', 'TP_Injury_Severity'
    ]
    X = df.drop(columns=drop_cols)
    
    # Verify model features for prohibited variables
    used_prohibited = [v for v in prohibited_vars if v in X.columns]
    
    # Preprocess data using the pipeline's preprocessor
    print("Preprocessing data for SHAP...")
    X_processed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Get feature names after one-hot encoding
    cat_features = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
    num_features = pipeline.named_steps['preprocessor'].transformers_[0][2]
    feature_names = np.concatenate([num_features, cat_features])
    
    # Create a DataFrame for SHAP with feature names
    X_processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed, columns=feature_names)
    
    # Explain the model using SHAP
    print("Calculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
    shap_values = explainer.shap_values(X_processed_df)
    
    # Generate Summary Plot
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_processed_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'shap_summary_plot.png'))
    plt.close()
    
    # Extract top 5 features by mean absolute SHAP value
    print("Identifying top cost drivers...")
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_5_drivers = feature_importance.head(5)
    
    # Create compliance report
    print("Writing compliance report...")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Model Explainability & Compliance Report\n\n")
        
        f.write("## 1. Compliance Check\n")
        if not used_prohibited:
            f.write("[PASS] No prohibited variables (Race, Religion, etc.) were found in the model features.\n\n")
        else:
            f.write("[WARNING] The following prohibited variables were found in the model features: " + ", ".join(used_prohibited) + "\n\n")
        
        f.write("## 2. Top 5 Cost Drivers (SHAP Analysis)\n")
        f.write("The following features have the most significant impact on predicting claim costs:\n\n")
        for i, row in top_5_drivers.iterrows():
            f.write(f"- **{row['col_name']}**: Average impact (SHAP value) of {row['feature_importance_vals']:.4f}\n")
        
        f.write("\n## 3. Summary Visualization\n")
        f.write("The SHAP summary plot visualizes how feature values influence the prediction (driving costs up or down). You can find the plot in: `reports/figures/shap_summary_plot.png`.\n")

    print(f"Report generated successfully at {report_path}")

if __name__ == '__main__':
    generate_explainability_report()
