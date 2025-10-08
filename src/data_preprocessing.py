# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import yaml

def preprocess_data(params_path):
    """
    Loads, preprocesses, and splits the data based on parameters.
    """
    print("Starting data preprocessing...")

    # Load parameters from the YAML file
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # --- Configuration ---
    RAW_DATA_PATH = r'data/raw/gender_pay_gap_india.csv'
    PROCESSED_DATA_FOLDER = 'data/processed'
    TARGET_COLUMN = 'Salary_LPA'
    TEST_SIZE = params['preprocess']['test_size']
    RANDOM_STATE = params['preprocess']['random_state']

    # Load data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    # Drop the ID column
    df = df.drop('ID', axis=1)

    # One-hot encode categorical features
    categorical_features = ['Gender', 'Education_Level', 'Job_Role', 'Region']
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