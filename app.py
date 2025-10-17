import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Create a FastAPI app instance
app = FastAPI(title="Customer Churn Prediction API")

# --- 1. Load the trained model ---
try:
    model = joblib.load("models/random_forest_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please run 'dvc repro' first.")
    model = None

# --- 2. Define the input data model (Pydantic model) ---
# This tells FastAPI what kind of data to expect in a request.
# The names MUST match the column names of your training data.
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # --- V V V V V V V V V V V V V V V V V V V --- #
    # TODO: ADD ALL OTHER COLUMNS FROM X_train.csv #
    # Example:                                      #
    # gender_Male: int = 0                           #
    # Partner_Yes: int = 0                          #
    # ... continue for all other one-hot encoded columns
    # --- ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ --- #


# --- 3. Create the prediction endpoint ---
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    """
    Receives customer data in JSON format and returns a churn prediction.
    """
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    # Convert the input data to a pandas DataFrame
    # The `dict()` method is deprecated, use `model_dump()` instead
    input_df = pd.DataFrame([customer_data.model_dump()])
    
    # IMPORTANT: Ensure the column order matches the training data.
    # We can rely on the Pydantic model to enforce the correct columns.

    # Make a prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Probability of churning

    # Return the result
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": f"{probability:.2%}"
    }

# --- 4. Add a root endpoint for a simple health check ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Churn Prediction API is running."}

# --- 5. Add a main block to run the app ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)