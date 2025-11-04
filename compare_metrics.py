import json
import sys

def compare_metrics(new_metrics_path, prod_metrics_path):
    """
    Compares new metrics with production metrics.
    Exits with 0 (success) if the new model is better OR equal.
    Exits with 1 (failure) only if the new model is worse.
    """
    with open(new_metrics_path, 'r') as f:
        new_metrics = json.load(f)
        
    with open(prod_metrics_path, 'r') as f:
        prod_metrics = json.load(f)

    # --- Define your "better model" criteria ---
    new_accuracy = new_metrics.get('accuracy', 0)
    prod_accuracy = prod_metrics.get('accuracy', 0)
    
    print(f"New Model Accuracy: {new_accuracy:.4f}")
    print(f"Production Model Accuracy: {prod_accuracy:.4f}")

    # --- THIS IS THE MODIFIED LOGIC ---
    if new_accuracy >= prod_accuracy:
        print("New model is better than or equal to production. Proceeding.")
        sys.exit(0) # Exit with a success code
    else:
        print("New model is WORSE than production. Skipping deployment.")
        sys.exit(1) # Exit with a failure code

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_metrics.py <new_metrics.json> <production_metrics.json>")
        sys.exit(2)
        
    compare_metrics(sys.argv[1], sys.argv[2])