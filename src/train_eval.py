"""
Watch Store Location Success Prediction - Model Training & Evaluation
Trains Logistic Regression & Random Forest Classifiers and calculates actual test metrics.
Saves model artifacts to models/trained_models.pkl.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from preprocess import prepare_train_test_data

def train_and_evaluate_models():
    data = prepare_train_test_data()
    
    X_train = data['X_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']

    # 1. Model 1: Logistic Regression (trained on scaled features)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, lr_preds),
        'Precision': precision_score(y_test, lr_preds),
        'Recall': recall_score(y_test, lr_preds),
        'F1-Score': f1_score(y_test, lr_preds),
        'ROC-AUC': roc_auc_score(y_test, lr_probs),
        'Confusion_Matrix': confusion_matrix(y_test, lr_preds).tolist()
    }

    # 2. Model 2: Random Forest Classifier (trained on original unscaled features)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'Model': 'Random Forest Classifier',
        'Accuracy': accuracy_score(y_test, rf_preds),
        'Precision': precision_score(y_test, rf_preds),
        'Recall': recall_score(y_test, rf_preds),
        'F1-Score': f1_score(y_test, rf_preds),
        'ROC-AUC': roc_auc_score(y_test, rf_probs),
        'Confusion_Matrix': confusion_matrix(y_test, rf_preds).tolist()
    }

    # 3. Create Comparison DataFrame
    metrics_df = pd.DataFrame([
        {
            'Model': lr_metrics['Model'],
            'Accuracy': f"{lr_metrics['Accuracy']:.4f}",
            'Precision': f"{lr_metrics['Precision']:.4f}",
            'Recall': f"{lr_metrics['Recall']:.4f}",
            'F1-Score': f"{lr_metrics['F1-Score']:.4f}",
            'ROC-AUC': f"{lr_metrics['ROC-AUC']:.4f}"
        },
        {
            'Model': rf_metrics['Model'],
            'Accuracy': f"{rf_metrics['Accuracy']:.4f}",
            'Precision': f"{rf_metrics['Precision']:.4f}",
            'Recall': f"{rf_metrics['Recall']:.4f}",
            'F1-Score': f"{rf_metrics['F1-Score']:.4f}",
            'ROC-AUC': f"{rf_metrics['ROC-AUC']:.4f}"
        }
    ])

    # 4. Model Selection based on primary metric (ROC-AUC)
    if rf_metrics['ROC-AUC'] >= lr_metrics['ROC-AUC']:
        selected_model_name = "Random Forest Classifier"
        selected_reason = f"Random Forest Classifier was selected for inference because it achieved the higher ROC-AUC ({rf_metrics['ROC-AUC']:.4f} vs {lr_metrics['ROC-AUC']:.4f}) on this test evaluation."
    else:
        selected_model_name = "Logistic Regression"
        selected_reason = f"Logistic Regression was selected for inference because it achieved the higher ROC-AUC ({lr_metrics['ROC-AUC']:.4f} vs {rf_metrics['ROC-AUC']:.4f}) on this test evaluation."

    # 5. Save trained models and scaler into top-level models/ directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_artifact_path = os.path.join(models_dir, "trained_models.pkl")
    
    with open(model_artifact_path, "wb") as f:
        pickle.dump({
            'lr_model': lr_model,
            'rf_model': rf_model,
            'scaler': scaler,
            'feature_names': data['feature_names'],
            'selected_model_name': selected_model_name
        }, f)

    return {
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'metrics_df': metrics_df,
        'selected_model_name': selected_model_name,
        'selected_reason': selected_reason,
        'model_artifact_path': model_artifact_path,
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }

if __name__ == "__main__":
    results = train_and_evaluate_models()
    print("\n==========================================")
    print(" ACTUAL MODEL EVALUATION RESULTS (TEST SET)")
    print("==========================================")
    print(results['metrics_df'].to_string(index=False))
    print(f"\nMODEL SELECTION: {results['selected_reason']}")
    print("\nConfusion Matrix - Logistic Regression (TN, FP / FN, TP):")
    print(np.array(results['lr_metrics']['Confusion_Matrix']))
    print("\nConfusion Matrix - Random Forest Classifier (TN, FP / FN, TP):")
    print(np.array(results['rf_metrics']['Confusion_Matrix']))
    print(f"\nModels successfully saved to: {results['model_artifact_path']}")
