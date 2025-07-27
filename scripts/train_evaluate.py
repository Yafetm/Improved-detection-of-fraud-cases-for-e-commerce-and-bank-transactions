import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_processed_data(data_path):
    """Load processed datasets."""
    fraud_train = pd.read_csv(os.path.join(data_path, 'processed_fraud_train.csv'))
    fraud_test = pd.read_csv(os.path.join(data_path, 'processed_fraud_test.csv'))
    creditcard_train = pd.read_csv(os.path.join(data_path, 'processed_creditcard_train.csv'))
    creditcard_test = pd.read_csv(os.path.join(data_path, 'processed_creditcard_test.csv'))
    return fraud_train, fraud_test, creditcard_train, creditcard_test

def train_model(model, X_train, y_train, X_test, y_test, model_name, dataset_name):
    """Train and evaluate a model, returning metrics and predictions."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    
    # Print classification report and confusion matrix
    print(f"\n{model_name} on {dataset_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\n{model_name} on {dataset_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, metrics, y_pred

def plot_feature_importance(model, X_columns, model_name, dataset_name):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({'Feature': X_columns, 'Importance': importance})
        feature_imp = feature_imp.sort_values('Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_imp)
        plt.title(f'Top 10 Feature Importance for {model_name} on {dataset_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_{dataset_name}_feature_importance.png')
        plt.close()

def main():
    # File paths
    data_path = os.path.join('data')
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    fraud_train, fraud_test, creditcard_train, creditcard_test = load_processed_data(data_path)
    
    # Prepare fraud data
    X_fraud_train = fraud_train.drop(['class', 'country'], axis=1, errors='ignore')  # Drop country and class
    y_fraud_train = fraud_train['class']
    X_fraud_test = fraud_test.drop(['class', 'country'], axis=1, errors='ignore')
    y_fraud_test = fraud_test['class']
    
    # Prepare creditcard data
    X_creditcard_train = creditcard_train.drop('Class', axis=1)
    y_creditcard_train = creditcard_train['Class']
    X_creditcard_test = creditcard_test.drop('Class', axis=1)
    y_creditcard_test = creditcard_test['Class']
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Train and evaluate models
    results = {'fraud_data': {}, 'creditcard_data': {}}
    
    for model_name, model in models.items():
        # Fraud data
        trained_model, metrics, _ = train_model(
            model, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test, model_name, 'fraud_data'
        )
        results['fraud_data'][model_name] = metrics
        plot_feature_importance(trained_model, X_fraud_train.columns, model_name, 'fraud_data')
        
        # Creditcard data
        trained_model, metrics, _ = train_model(
            model, X_creditcard_train, y_creditcard_train, X_creditcard_test, y_creditcard_test, model_name, 'creditcard_data'
        )
        results['creditcard_data'][model_name] = metrics
        plot_feature_importance(trained_model, X_creditcard_train.columns, model_name, 'creditcard_data')
    
    # Save results to a CSV
    results_df = pd.DataFrame({
        'Dataset': [],
        'Model': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC-AUC': []
    })
    
    for dataset, model_results in results.items():
        for model_name, metrics in model_results.items():
            results_df = pd.concat([results_df, pd.DataFrame({
                'Dataset': [dataset],
                'Model': [model_name],
                'Precision': [metrics['precision']],
                'Recall': [metrics['recall']],
                'F1-Score': [metrics['f1_score']],
                'ROC-AUC': [metrics['roc_auc']]
            })], ignore_index=True)
    
    results_df.to_csv('results/model_performance.csv', index=False)
    print("\nModel performance metrics saved to results/model_performance.csv")

if __name__ == "__main__":
    main()