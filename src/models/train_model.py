import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import joblib
import os

def train_model():
    # Define paths
    processed_dir = 'data/processed'
    models_dir = 'src/models'
    
    # Load data
    print("Loading ABT...")
    df = pd.read_csv(os.path.join(processed_dir, 'abt.csv'))
    
    # Define features and target
    # Target is Log_Ultimate_Claim_Amount
    target = 'Log_Ultimate_Claim_Amount'
    original_target = 'Ultimate_Claim_Amount'
    
    # Features (excluding IDs, dates, and target variables)
    drop_cols = [
        'Claim_ID', 'Policy_ID', 'Customer_ID', 'Accident_Date', 'FNOL_Date',
        'Ultimate_Claim_Amount', 'Log_Ultimate_Claim_Amount', 'Status',
        'TP_ID', 'TP_Injury_Severity' # TP_Injury_Severity replaced by numeric score
    ]
    
    X = df.drop(columns=drop_cols)
    y = df[target]
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numeric features: {numeric_cols}")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Create XGBoost Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ))
    ])
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Evaluating model...")
    y_pred_log = model.predict(X_test)
    
    # Transform back to original scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y_test_orig, y_pred_orig)
    mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")
    
    # Save model
    model_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Return metrics for report
    return rmse, mape

if __name__ == '__main__':
    train_model()
