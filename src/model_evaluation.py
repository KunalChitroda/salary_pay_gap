# src/model_evaluation.py

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# --- Configuration ---
PROCESSED_DATA_FOLDER = 'data/processed'
MODEL_FOLDER = 'models'
MODEL_NAME = 'random_forest_model.pkl'

def evaluate_model():
    """
    Loads the test data and the trained model, then evaluates the model's performance.
    """
    print("Starting model evaluation...")

    # Load the model
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ Error: Model '{model_path}' not found. Please train the model first.")
        return

    # Load the test data
    try:
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv')).values.ravel()
    except FileNotFoundError:
        print("❌ Error: Test data not found. Please run the data preprocessing script first.")
        return

    print(f"Loaded {len(X_test)} test samples.")

    # Make predictions on the test data
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("         Model Performance Metrics")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  -> On average, the model's predictions are off by ~₹{mae:.2f} Lakhs.")
    print("-" * 50)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2%}")
    print(f"  -> The model explains {r2:.2%} of the variance in the salary data.")
    print("="*50)

if __name__ == '__main__':
    evaluate_model()