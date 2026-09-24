"""
Watch Store Location Success Prediction - Data Preprocessor
Handles loading data, checking for missing values and duplicate rows,
feature/target separation, feature scaling, and stratified train-test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'population',
    'average_monthly_income',
    'daily_foot_traffic',
    'nearby_competitors',
    'monthly_rent',
    'distance_to_mall_km',
    'nearby_retail_stores',
    'estimated_monthly_customers',
    'average_purchase_value',
    'monthly_operating_cost',
    'local_demand_score',
    'target_age_group_score'
]

TARGET_COLUMN = 'profitable'

def load_and_inspect_data(csv_path=None):
    """
    Loads dataset and checks for missing values and duplicate rows.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "watch_store_locations.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_dataset.py first.")

    df = pd.read_csv(csv_path)
    
    quality_summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'class_counts': df[TARGET_COLUMN].value_counts().to_dict(),
        'class_proportions': df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    }
    
    return df, quality_summary

def prepare_train_test_data(csv_path=None, test_size=0.2, random_state=42):
    """
    Prepares features and target using an 80/20 stratified split.
    Fits StandardScaler ONLY on X_train and applies transform to X_test to prevent data leakage.
    """
    df, quality_summary = load_and_inspect_data(csv_path)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling for Logistic Regression (fitted ONLY on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)
    
    return {
        'df': df,
        'quality_summary': quality_summary,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled_df,
        'X_test_scaled': X_test_scaled_df,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': FEATURE_COLUMNS
    }

if __name__ == "__main__":
    data = prepare_train_test_data()
    print("Data Preprocessing Check Completed Successfully.")
    print(f"Dataset Shape: {data['quality_summary']['shape']}")
    print(f"Missing Values Check: {data['quality_summary']['missing_values']}")
    print(f"Duplicate Rows Check: {data['quality_summary']['duplicate_rows']}")
    print(f"Training Set Size: {len(data['X_train'])}")
    print(f"Testing Set Size: {len(data['X_test'])}")
