import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

def load_data(data_path):
    """Load raw datasets from the data folder."""
    fraud_data = pd.read_csv(os.path.join(data_path, 'Fraud_Data.csv'))
    ip_data = pd.read_csv(os.path.join(data_path, 'IpAddress_to_Country.csv'))
    creditcard_data = pd.read_csv(os.path.join(data_path, 'creditcard.csv'))
    return fraud_data, ip_data, creditcard_data

def check_missing_values(df, name):
    """Check and handle missing values."""
    print(f"\nMissing values in {name}:\n{df.isnull().sum()}")
    df = df.dropna()  # Drop rows with any missing values
    return df

def clean_data(df, name):
    """Remove duplicates and correct data types."""
    print(f"\nDuplicates in {name}: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    
    if name == 'fraud_data':
        # Try multiple datetime formats
        for col in ['signup_time', 'purchase_time']:
            try:
                print(f"Sample {col} before conversion:\n{df[col].head()}")
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            except Exception as e:
                print(f"Error converting {col}: {e}")
            if df[col].isna().any():
                print(f"Warning: NaT values in {col}. Dropping affected rows.")
                df = df.dropna(subset=[col])
        df['ip_address'] = df['ip_address'].astype(int)
        df[['source', 'browser', 'sex']] = df[['source', 'browser', 'sex']].astype('category')
    elif name == 'creditcard_data':
        df['Amount'] = df['Amount'].astype(float)
        df['Class'] = df['Class'].astype(int)
        for col in [f'V{i}' for i in range(1, 29)]:
            df[col] = df[col].astype(float)
    
    print(f"Data types in {name} after cleaning:\n{df.dtypes}")
    return df

def merge_ip_to_country(fraud_data, ip_data):
    """Merge IP addresses to country using vectorized operations."""
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    
    # Ensure datetime types before merge
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], infer_datetime_format=True, errors='coerce')
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], infer_datetime_format=True, errors='coerce')
    
    if fraud_data['signup_time'].isna().any() or fraud_data['purchase_time'].isna().any():
        print("Warning: NaT values in datetime columns before merge. Dropping affected rows.")
        fraud_data = fraud_data.dropna(subset=['signup_time', 'purchase_time'])
    
    # Sort for vectorized merge
    fraud_data = fraud_data.sort_values('ip_address')
    ip_data = ip_data.sort_values('lower_bound_ip_address')
    
    # Perform merge_asof
    merged = pd.merge_asof(
        fraud_data,
        ip_data[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter valid IP ranges
    merged = merged[
        (merged['ip_address'] >= merged['lower_bound_ip_address']) & 
        (merged['ip_address'] <= merged['upper_bound_ip_address'])
    ]
    
    # Fill unmatched countries
    merged['country'] = merged['country'].fillna('Unknown')
    merged = merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)
    
    # Ensure datetime types after merge
    merged['signup_time'] = pd.to_datetime(merged['signup_time'], infer_datetime_format=True, errors='coerce')
    merged['purchase_time'] = pd.to_datetime(merged['purchase_time'], infer_datetime_format=True, errors='coerce')
    
    if merged['signup_time'].isna().any() or merged['purchase_time'].isna().any():
        print("Warning: NaT values in datetime columns after merge. Dropping affected rows.")
        merged = merged.dropna(subset=['signup_time', 'purchase_time'])
    
    print("Data types after IP merge:\n", merged.dtypes)
    return merged

def engineer_features(fraud_data):
    """Create new features for fraud_data."""
    if not (pd.api.types.is_datetime64_any_dtype(fraud_data['signup_time']) and 
            pd.api.types.is_datetime64_any_dtype(fraud_data['purchase_time'])):
        raise ValueError("Datetime columns must be in datetime64 format")
    
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600.0
    
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
    fraud_data['avg_purchase_value'] = fraud_data.groupby('user_id')['purchase_value'].transform('mean')
    
    print("Data types after feature engineering:\n", fraud_data.dtypes)
    return fraud_data

def encode_and_scale(fraud_data, creditcard_data):
    """Encode categorical features, scale numerical features, and drop datetime columns."""
    # Encode fraud_data categorical features
    categorical_cols = ['source', 'browser', 'sex', 'country']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(fraud_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical and datetime columns
    fraud_data = fraud_data.drop(categorical_cols + ['signup_time', 'purchase_time'], axis=1).reset_index(drop=True)
    fraud_data = pd.concat([fraud_data, encoded_df], axis=1)
    
    # Scale numerical features
    scaler = StandardScaler()
    fraud_numerical = ['purchase_value', 'age', 'time_since_signup', 'transaction_count', 'avg_purchase_value']
    fraud_data[fraud_numerical] = scaler.fit_transform(fraud_data[fraud_numerical])
    
    creditcard_numerical = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    creditcard_data[creditcard_numerical] = scaler.fit_transform(creditcard_data[creditcard_numerical])
    
    print("Data types after encoding and scaling (fraud_data):\n", fraud_data.dtypes)
    return fraud_data, creditcard_data

def apply_smote(X, y, name):
    """Apply SMOTE to training data after train-test split."""
    print(f"\nClass distribution in {name}:\n", y.value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution in {name} (training):\n", pd.Series(y_train_resampled).value_counts())
    return X_train_resampled, y_train_resampled, X_test, y_test

def main():
    data_path = os.path.join('data')
    fraud_data, ip_data, creditcard_data = load_data(data_path)
    
    # Handle missing values
    fraud_data = check_missing_values(fraud_data, 'fraud_data')
    creditcard_data = check_missing_values(creditcard_data, 'creditcard_data')
    
    # Clean data
    fraud_data = clean_data(fraud_data, 'fraud_data')
    creditcard_data = clean_data(creditcard_data, 'creditcard_data')
    
    # Merge IP data
    fraud_data = merge_ip_to_country(fraud_data, ip_data)
    
    # Feature engineering
    fraud_data = engineer_features(fraud_data)
    
    # Encode and scale
    fraud_data, creditcard_data = encode_and_scale(fraud_data, creditcard_data)
    
    # Prepare features and labels for fraud data
    X_fraud = fraud_data.drop('class', axis=1)

    # --- FIX: keep only numeric columns before SMOTE ---
    X_fraud = X_fraud.select_dtypes(include=[np.number])

    y_fraud = fraud_data['class']
    X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test = apply_smote(X_fraud, y_fraud, 'fraud_data')
    
    X_creditcard = creditcard_data.drop('Class', axis=1)
    y_creditcard = creditcard_data['Class']
    X_creditcard_train, y_creditcard_train, X_creditcard_test, y_creditcard_test = apply_smote(X_creditcard, y_creditcard, 'creditcard_data')
    
    # Save processed data
    fraud_train = pd.concat([pd.DataFrame(X_fraud_train, columns=X_fraud.columns), y_fraud_train.rename('class')], axis=1)
    fraud_test = pd.concat([pd.DataFrame(X_fraud_test, columns=X_fraud.columns), y_fraud_test.rename('class')], axis=1)
    creditcard_train = pd.concat([pd.DataFrame(X_creditcard_train, columns=X_creditcard.columns), y_creditcard_train.rename('Class')], axis=1)
    creditcard_test = pd.concat([pd.DataFrame(X_creditcard_test, columns=X_creditcard.columns), y_creditcard_test.rename('Class')], axis=1)
    
    fraud_train.to_csv(os.path.join(data_path, 'processed_fraud_train.csv'), index=False)
    fraud_test.to_csv(os.path.join(data_path, 'processed_fraud_test.csv'), index=False)
    creditcard_train.to_csv(os.path.join(data_path, 'processed_creditcard_train.csv'), index=False)
    creditcard_test.to_csv(os.path.join(data_path, 'processed_creditcard_test.csv'), index=False)
    
    print("\nProcessed datasets saved to data/ folder.")

if __name__ == "__main__":
    main()
