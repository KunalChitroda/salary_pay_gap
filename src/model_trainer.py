# src/model_trainer.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import sys
import yaml # Import yaml
import mlflow
import mlflow.sklearn

def train_model(params_path):
    """
    Loads data and fixed params, then trains and saves the model.
    """
    print("Starting model training with fixed parameters...")

    # Load parameters from the YAML file
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # --- Configuration ---
    PROCESSED_DATA_FOLDER = 'data/processed'
    MODEL_FOLDER = 'models'
    MODEL_NAME = 'random_forest_model.pkl'
    # Get training parameters from the 'train' section of params.yaml
    train_params = params['train']
    RANDOM_STATE = params['preprocess']['random_state']

    with mlflow.start_run():
        print(f"Using parameters: {train_params}")
        mlflow.log_params(train_params) # Log the fixed parameters

        # Load the processed training data
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv')).values.ravel()

        # Initialize the model with the fixed parameters
        model = RandomForestRegressor(**train_params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        print("✅ Model training complete.")

        # Save the model
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
        joblib.dump(model, model_path)
        print(f"Model saved to '{model_path}'")

if __name__ == '__main__':
    train_model(params_path=sys.argv[1])