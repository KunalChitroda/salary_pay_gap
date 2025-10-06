# main.py

# Imports from the 'src' folder
from src.data_preprocessing import preprocess_data
from src.model_trainer import train_model
from src.model_evaluation import evaluate_model

# --- Configuration ---
RAW_DATA_PATH = r'data\raw\gender_pay_gap_india.csv'

def run_pipeline():
    """
    Executes the entire machine learning pipeline from start to finish.
    """
    print("🚀 Starting the full machine learning pipeline...")

    # Step 1: Preprocess the data
    print("\n--- Step 1: Data Preprocessing ---")
    preprocess_data(RAW_DATA_PATH)

    # Step 2: Train the model
    print("\n--- Step 2: Model Training ---")
    train_model()

    # Step 3: Evaluate the model
    print("\n--- Step 3: Model Evaluation ---")
    evaluate_model()

    print("\n✅ Pipeline execution finished successfully!")

if __name__ == '__main__':
    run_pipeline()