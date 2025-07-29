import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_processed_data(data_path):
    """Load processed datasets."""
    fraud_train = pd.read_csv(os.path.join(data_path, 'processed_fraud_train.csv'))
    fraud_test = pd.read_csv(os.path.join(data_path, 'processed_fraud_test.csv'))
    creditcard_train = pd.read_csv(os.path.join(data_path, 'processed_creditcard_train.csv'))
    creditcard_test = pd.read_csv(os.path.join(data_path, 'processed_creditcard_test.csv'))
    return fraud_train, fraud_test, creditcard_train, creditcard_test

def tune_xgboost(X_train, y_train, X_test, y_test, dataset_name):
    """Tune XGBoost hyperparameters using GridSearchCV."""
    param_grid = {
        'xgb__n_estimators': [50, 100],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.01, 0.1]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters for {dataset_name}: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\nTuned XGBoost on {dataset_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nTuned XGBoost on {dataset_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model, metrics

def save_model(model, dataset_name):
    """Save the trained model."""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/tuned_xgboost_{dataset_name}.joblib')
    print(f"\nSaved tuned XGBoost model for {dataset_name} to models/tuned_xgboost_{dataset_name}.joblib")

def create_prediction_pipeline():
    """Create a prediction pipeline for new fraud_data."""
    # Define preprocessing steps
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'transaction_count', 'avg_purchase_value']
    categorical_cols = ['source', 'browser', 'sex', 'country']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    return pipeline

def main():
    # File paths
    data_path = os.path.join('data')
    os.makedirs('results', exist_ok=True)
    
    # Load data
    fraud_train, fraud_test, creditcard_train, creditcard_test = load_processed_data(data_path)
    
    # Prepare fraud data
    X_fraud_train = fraud_train.drop(['class', 'country', 'device_id'], axis=1, errors='ignore')
    y_fraud_train = fraud_train['class']
    X_fraud_test = fraud_test.drop(['class', 'country', 'device_id'], axis=1, errors='ignore')
    y_fraud_test = fraud_test['class']
    
    # Prepare creditcard data
    X_creditcard_train = creditcard_train.drop('Class', axis=1)
    y_creditcard_train = creditcard_train['Class']
    X_creditcard_test = creditcard_test.drop('Class', axis=1)
    y_creditcard_test = creditcard_test['Class']
    
    # Tune and evaluate XGBoost
    print("\nTuning XGBoost for fraud_data...")
    fraud_model, fraud_metrics = tune_xgboost(X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test, 'fraud_data')
    
    print("\nTuning XGBoost for creditcard_data...")
    creditcard_model, creditcard_metrics = tune_xgboost(X_creditcard_train, y_creditcard_train, X_creditcard_test, y_creditcard_test, 'creditcard_data')
    
    # Save models
    save_model(fraud_model, 'fraud_data')
    save_model(creditcard_model, 'creditcard_data')
    
    # Save tuned metrics
    results_df = pd.DataFrame({
        'Dataset': ['fraud_data', 'creditcard_data'],
        'Model': ['Tuned_XGBoost', 'Tuned_XGBoost'],
        'Precision': [fraud_metrics['precision'], creditcard_metrics['precision']],
        'Recall': [fraud_metrics['recall'], creditcard_metrics['recall']],
        'F1-Score': [fraud_metrics['f1_score'], creditcard_metrics['f1_score']],
        'ROC-AUC': [fraud_metrics['roc_auc'], creditcard_metrics['roc_auc']]
    })
    
    results_df.to_csv('results/tuned_model_performance.csv', index=False)
    print("\nTuned model performance metrics saved to results/tuned_model_performance.csv")
    
    # Create and save prediction pipeline (for fraud_data)
    pipeline = create_prediction_pipeline()
    joblib.dump(pipeline, 'models/fraud_prediction_pipeline.joblib')
    print("\nSaved fraud prediction pipeline to models/fraud_prediction_pipeline.joblib")

if __name__ == "__main__":
    main()