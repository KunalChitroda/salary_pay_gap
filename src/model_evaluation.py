# src/model_evaluation.py

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import json
import mlflow  # Import mlflow

# --- Configuration ---
PROCESSED_DATA_FOLDER = 'data/processed'
MODEL_FOLDER = 'models'
MODEL_NAME = 'random_forest_model.pkl'
METRICS_FILE = 'metrics.json'
PARAMS_FILE = 'best_params.json' # Add path to params file

def evaluate_model():
    """
    Evaluates the model and logs parameters and metrics to MLflow.
    """
    print("Starting model evaluation and MLflow logging...")

    # Start a new MLflow run for evaluation
    with mlflow.start_run():
        # Load the model and test data
        model = joblib.load(os.path.join(MODEL_FOLDER, MODEL_NAME))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv')).values.ravel()

        # Load the parameters that were used to train this model
        with open(PARAMS_FILE) as f:
            params = json.load(f)
        
        # --- MLFLOW: Log the parameters ---
        print(f"Logging parameters: {params}")
        mlflow.log_params(params)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # --- MLFLOW: Log the metrics ---
        print(f"Logging metrics: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2%}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Save metrics to JSON file for DVC
        with open(METRICS_FILE, 'w') as f:
            json.dump({"mae": mae, "mse": mse, "r2_score": r2}, f, indent=4)
        print(f"Metrics saved to '{METRICS_FILE}' for DVC.")

if __name__ == '__main__':
    evaluate_model()