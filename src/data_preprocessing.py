import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import yaml

def preprocess_data(params_path):
    """
    Loads and preprocesses the Telco Customer Churn dataset.
    """
    print("Starting data preprocessing for Customer Churn...")

    # Load parameters from the YAML file
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # --- Configuration ---
    # Corrected the RAW_DATA_PATH to the new file name
    RAW_DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
    PROCESSED_DATA_FOLDER = 'data/processed'
    # Corrected the TARGET_COLUMN
    TARGET_COLUMN = 'Churn'
    TEST_SIZE = params['preprocess']['test_size']
    RANDOM_STATE = params['preprocess']['random_state']

    # Load data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    # --- Data Cleaning and Feature Engineering ---

    # Drop the customerID column as it's not a predictive feature
    df = df.drop('customerID', axis=1)
    print("Dropped 'customerID' column.")

    # The 'TotalCharges' column is an object type and contains spaces.
    # Convert it to a numeric type, coercing errors to NaN (Not a Number).
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop the few rows that now have missing values for TotalCharges.
    df.dropna(inplace=True)
    print("Cleaned 'TotalCharges' column.")

    # Convert the target variable 'Churn' from "Yes"/"No" to 1/0.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)
    print("Encoded target column 'Churn' to 1/0.")

    # One-hot encode all other categorical features
    # This list contains all non-numeric columns except the target.
    categorical_features = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    print("One-hot encoded categorical features.")

    # Separate features (X) and target (y)
    X = df_encoded.drop(TARGET_COLUMN, axis=1)
    y = df_encoded[TARGET_COLUMN]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # Create processed data folder if it doesn't exist
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    # Save the processed data
    X_train.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv'), index=False)
    print(f"Processed data saved to the '{PROCESSED_DATA_FOLDER}' directory.")

if __name__ == '__main__':
    # DVC will pass 'params.yaml' as the first command-line argument
    preprocess_data(params_path=sys.argv[1])