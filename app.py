import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Optional
import json  # Import json
import shap  # Import shap

# Create a FastAPI app instance
app = FastAPI(title="Customer Churn Prediction API")

# --- 1. Load all model assets on startup ---
try:
    model = joblib.load("models/random_forest_model.pkl")
    
    # Load the column list saved during training
    with open("models/model_columns.json", 'r') as f:
        model_columns = json.load(f)
    
    # Create the SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    print("Model, columns, and SHAP explainer loaded successfully.")
except FileNotFoundError:
    print("Error: Model assets (model.pkl or model_columns.json) not found. Please run 'dvc repro' first.")
    model = None
    model_columns = None
    explainer = None
except Exception as e:
    print(f"Error loading model assets: {e}")
    model = None
    model_columns = None
    explainer = None


# --- 2. Define the input data model (Pydantic model) ---
class CustomerData(BaseModel):
    SeniorCitizen: int = 0
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int = 0
    Partner_Yes: int = 0
    Dependents_Yes: int = 0
    PhoneService_Yes: int = 0
    MultipleLines_No_phone_service: int = 0
    MultipleLines_Yes: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    OnlineSecurity_No_internet_service: int = 0
    OnlineSecurity_Yes: int = 0
    OnlineBackup_No_internet_service: int = 0
    OnlineBackup_Yes: int = 0
    DeviceProtection_No_internet_service: int = 0
    DeviceProtection_Yes: int = 0
    TechSupport_No_internet_service: int = 0
    TechSupport_Yes: int = 0
    StreamingTV_No_internet_service: int = 0
    StreamingTV_Yes: int = 0
    StreamingMovies_No_internet_service: int = 0
    StreamingMovies_Yes: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaperlessBilling_Yes: int = 0
    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0


# --- 3. Create the prediction endpoint ---
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    """
    Receives customer data, renames columns, and returns a churn prediction
    along with SHAP values for explainability.
    """
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    input_df = pd.DataFrame([customer_data.model_dump()])

    # --- Column Renaming ---
    column_mapping = {
        'MultipleLines_No_phone_service': 'MultipleLines_No phone service',
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'InternetService_No': 'InternetService_No',
        'OnlineSecurity_No_internet_service': 'OnlineSecurity_No internet service',
        'OnlineBackup_No_internet_service': 'OnlineBackup_No internet service',
        'DeviceProtection_No_internet_service': 'DeviceProtection_No internet service',
        'TechSupport_No_internet_service': 'TechSupport_No internet service',
        'StreamingTV_No_internet_service': 'StreamingTV_No internet service',
        'StreamingMovies_No_internet_service': 'StreamingMovies_No internet service',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year',
        'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check'
    }
    try:
        input_df_renamed = input_df.rename(columns=column_mapping)
    except Exception as e:
         return {"error": f"Error renaming columns: {e}"}

    # --- Ensure column order matches training ---
    try:
        # Use the column list loaded on startup
        input_df_final = input_df_renamed.reindex(columns=model_columns, fill_value=0)
    except Exception as e:
        return {"error": f"Error reordering columns: {e}. Expected columns: {model_columns}, Got columns: {input_df_renamed.columns.tolist()}"}

    # --- Make prediction and get explanation ---
    try:
        prediction = model.predict(input_df_final)[0]
        probability = model.predict_proba(input_df_final)[0][1]
        
        # Calculate SHAP values for this one prediction
        shap_values_obj = explainer.shap_values(input_df_final)
        
        # Get values for class 1 (Churn)
        shap_values = shap_values_obj[1][0]
        base_value = explainer.expected_value[1]

    except Exception as e:
        return {"error": f"Prediction or SHAP error: {e}. Input columns: {input_df_final.columns.tolist()}"}

    # --- Return all data ---
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": f"{probability:.2%}",
        "shap_base_value": base_value,
        "shap_values": shap_values.tolist(), # Convert numpy array to list
        "feature_names": model_columns, # Send feature names
        "feature_values": input_df_final.iloc[0].tolist() # Send feature values
    }

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Churn Prediction API is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)