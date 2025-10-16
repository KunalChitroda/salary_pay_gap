# Imports from the 'src' folder
from src.data_preprocessing import preprocess_data
from src.model_trainer import train_model
from src.model_evaluation import evaluate_model

# --- Configuration ---
PARAMS_PATH = 'params.yaml'

def run_pipeline():
    """
    Executes the entire machine learning pipeline from start to finish
    by calling each script with the parameters file.
    """
    print("🚀 Starting the full machine learning pipeline for Customer Churn...")

    # Step 1: Preprocess the data
    print("\n--- Step 1: Data Preprocessing ---")
    # Pass the params file to the function
    preprocess_data(params_path=PARAMS_PATH)

    # Step 2: Train the model
    print("\n--- Step 2: Model Training ---")
    # Pass the params file to the function
    train_model(params_path=PARAMS_PATH)

    # Step 3: Evaluate the model
    print("\n--- Step 3: Model Evaluation ---")
    # Pass the params file to the function
    evaluate_model(params_path=PARAMS_PATH)

    print("\n✅ Pipeline execution finished successfully!")

if __name__ == '__main__':
    run_pipeline()