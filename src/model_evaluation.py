import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
import yaml
import json
import mlflow
import shap               # Import SHAP
import matplotlib.pyplot as plt # Import matplotlib

# --- Configuration ---
PROCESSED_DATA_FOLDER = 'data/processed'
MODEL_FOLDER = 'models'
MODEL_NAME = 'random_forest_model.pkl'
METRICS_FILE = 'metrics.json'

def evaluate_model(params_path):
    """
    Evaluates the classification model, logs parameters, metrics,
    and a SHAP summary plot to MLflow.
    """
    print("Starting model evaluation for Customer Churn...")

    # Load parameters from the main YAML file
    with open(params_path) as f:
        params = yaml.safe_load(f)

    train_params = params['train']

    # Start a new MLflow run for evaluation
    with mlflow.start_run():
        # Load the model and test data
        model = joblib.load(os.path.join(MODEL_FOLDER, MODEL_NAME))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv')).values.ravel()
        
        # --- MLFLOW: Log the parameters ---
        print(f"Logging parameters: {train_params}")
        mlflow.log_params(train_params)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # --- MLFLOW: Log the metrics ---
        print(f"Logging metrics: Accuracy={accuracy:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save metrics to JSON file for DVC
        with open(METRICS_FILE, 'w') as f:
            json.dump({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }, f, indent=4)
        print(f"Metrics saved to '{METRICS_FILE}' for DVC.")

        # --- UPDATED SHAP SECTION ---
        print("Calculating SHAP summary plot...")
        if len(X_test) > 100:
            X_test_sample = X_test.sample(100, random_state=42)
        else:
            X_test_sample = X_test
        
        # Create the explainer
        explainer = shap.TreeExplainer(model)
        
        # Create the new, robust Explanation object
        explanation = explainer(X_test_sample)
        
        # Create and save the summary plot
        # We slice the Explanation object: [:, :, 1] means (all_samples, all_features, class_1)
        shap.summary_plot(explanation[:, :, 1], X_test_sample, show=False)
        
        plot_path = "shap_summary_plot.png"
        plt.savefig(plot_path, bbox_inches='tight') # Save the plot
        plt.close() # Close the plot to free up memory
        
        # Log the plot as an artifact in MLflow
        mlflow.log_artifact(plot_path)
        print(f"SHAP summary plot saved and logged to MLflow as '{plot_path}'.")
        # --- END OF UPDATED SHAP SECTION ---

if __name__ == '__main__':
    evaluate_model(params_path='params.yaml')