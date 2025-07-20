import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from datetime import datetime
import os

def load_data(fraud_path, ip_path, creditcard_path):
    """Load the datasets."""
    fraud_data = pd.read_csv(fraud_path)
    ip_data = pd.read_csv(ip_path)
    creditcard_data = pd.read_csv(creditcard_path)
    return fraud_data, ip_data, creditcard_data

def handle_missing_values(df, dataset_name):
    """Handle missing values by imputing or dropping."""
    print(f"\nMissing values in {dataset_name}:\n", df.isnull().sum())
    # For simplicity, drop rows with missing values (modify based on EDA insights)
    df = df.dropna()
    return df

def clean_data(df, dataset_name):
    """Remove duplicates and correct data types."""
    print(f"\nDuplicates in {dataset_name}: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    # Ensure correct data types for fraud_data
    if dataset_name == 'fraud_data':
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def merge_ip_data(fraud_data, ip_data):
    """Merge fraud_data with ip_data based on IP address ranges."""
    # Convert IP addresses to integer
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)
    
    # Function to map IP to country
    def map_ip_to_country(ip, ip_data):
        for _, row in ip_data.iterrows():
            if row['lower_bound_ip_address'] <= ip <= row['upper_bound_ip_address']:
                return row['country']
        return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(lambda x: map_ip_to_country(x, ip_data))
    return fraud_data

def feature_engineering(fraud_data):
    """Engineer features for fraud_data."""
    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600.0  # Hours
    
    # Transaction frequency and velocity (example: count transactions per user)
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
    fraud_data['avg_purchase_value'] = fraud_data.groupby('user_id')['purchase_value'].transform('mean')
    return fraud_data

def encode_categorical(fraud_data):
    """Encode categorical features."""
    categorical_cols = ['source', 'browser', 'sex', 'country']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(fraud_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns and concatenate encoded ones
    fraud_data = fraud_data.drop(categorical_cols, axis=1).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    return pd.concat([fraud_data, encoded_df], axis=1)

def scale_features(fraud_data, creditcard_data):
    """Scale numerical features."""
    scaler = StandardScaler()
    # Fraud data numerical columns
    

    fraud_numerical = ['purchase_value', 'age', 'time_since_signup', 'transaction_count', 'avg_purchase_value']
    fraud_data[fraud_numerical] = scaler.fit_transform(fraud_data[fraud_numerical])
    
    # Creditcard data numerical columns
    creditcard_numerical = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    creditcard_data[creditcard_numerical] = scaler.fit_transform(creditcard_data[creditcard_numerical])
    
    return fraud_data, creditcard_data

def handle_class_imbalance(X, y, dataset_name):
    """Apply SMOTE to handle class imbalance."""
    print(f"\nClass distribution in {dataset_name}:\n", y.value_counts())
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def main():
    # File paths
    data_path = os.path.join('..', 'data')
    fraud_path = os.path.join(data_path, 'Fraud_Data.csv')
    ip_path = os.path.join(data_path, 'IpAddress_to_Country.csv')
    creditcard_path = os.path.join(data_path, 'creditcard.csv')

    
    # Load data
    fraud_data, ip_data, creditcard_data = load_data(fraud_path, ip_path, creditcard_path)
    
    # Handle missing values
    fraud_data = handle_missing_values(fraud_data, 'fraud_data')
    creditcard_data = handle_missing_values(creditcard_data, 'creditcard_data')
    
    # Clean data
    fraud_data = clean_data(fraud_data, 'fraud_data')
    creditcard_data = clean_data(creditcard_data, 'creditcard_data')
    
    # Merge IP data
    fraud_data = merge_ip_data(fraud_data, ip_data)
    
    # Feature engineering
    fraud_data = feature_engineering(fraud_data)
    
    # Encode categorical features
    fraud_data = encode_categorical(fraud_data)
    
    # Scale features
    fraud_data, creditcard_data = scale_features(fraud_data, creditcard_data)
    
    # Handle class imbalance
    X_fraud = fraud_data.drop('class', axis=1)
    y_fraud = fraud_data['class']
    X_fraud_resampled, y_fraud_resampled = handle_class_imbalance(X_fraud, y_fraud, 'fraud_data')
    
    X_creditcard = creditcard_data.drop('Class', axis=1)
    y_creditcard = creditcard_data['Class']
    X_creditcard_resampled, y_creditcard_resampled = handle_class_imbalance(X_creditcard, y_creditcard, 'creditcard_data')
    
    # Save processed data
    fraud_data_resampled = pd.concat([pd.DataFrame(X_fraud_resampled, columns=X_fraud.columns), y_fraud_resampled], axis=1)
    creditcard_data_resampled = pd.concat([pd.DataFrame(X_creditcard_resampled, columns=X_creditcard.columns), y_creditcard_resampled], axis=1)
    
    fraud_data_resampled.to_csv(os.path.join(data_path, 'processed_fraud_data.csv'), index=False)
    creditcard_data_resampled.to_csv(os.path.join(data_path, 'processed_creditcard_data.csv'), index=False)
    print("\nProcessed data saved to data/ folder.")

if __name__ == "__main__":
    main() 
