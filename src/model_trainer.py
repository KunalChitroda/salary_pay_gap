# src/model_trainer.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json

# --- Configuration ---
PROCESSED_DATA_FOLDER = 'data/processed'
MODEL_FOLDER = 'models'
MODEL_NAME = 'random_forest_model.pkl'
PARAMS_FILE = 'best_params.json'

def train_model():
    """
    Loads data and tuned params, then trains and saves the model.
    (MLflow logging has been removed from this script).
    """
    print("Starting model training...")

    # Load the best parameters from the tuning stage
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    print(f"Loaded best parameters: {params}")

    # Load the processed training data
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv')).values.ravel()

    # Initialize the model with the best parameters
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Save the model with joblib for the DVC pipeline
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    joblib.dump(model, model_path)
    print(f"Model saved to '{model_path}'")

if __name__ == '__main__':
    train_model()