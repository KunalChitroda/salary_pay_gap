import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Optional
import json
import shap

# Create a FastAPI app instance
app = FastAPI(title="Customer Churn Prediction API")

# --- 1. Load all model assets on startup ---
try:
    model = joblib.load("models/random_forest_model.pkl")
    with open("models/model_columns.json", 'r') as f:
        model_columns = json.load(f)
    explainer = shap.TreeExplainer(model)
    print("Model, columns, and SHAP explainer loaded successfully.")
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
        input_df_final = input_df_renamed.reindex(columns=model_columns, fill_value=0)
    except Exception as e:
        return {"error": f"Error reordering columns: {e}."}

    # --- Make prediction and get explanation ---
    try:
        prediction = model.predict(input_df_final)[0]
        probability = model.predict_proba(input_df_final)[0][1]
        
        # --- V V V V V THE FIX V V V V V ---
        # Use the modern explainer(X) syntax
        explanation = explainer(input_df_final)
        
        # Slice the Explanation object to get values for class 1 (Churn)
        # syntax: [sample_index, feature_index, class_index]
        shap_values = explanation[0, :, 1].values
        base_value = explanation[0, :, 1].base_values
        # --- ^ ^ ^ ^ ^ THE FIX ^ ^ ^ ^ ^ ---

    except Exception as e:
        return {"error": f"Prediction or SHAP error: {e}. Input columns: {input_df_final.columns.tolist()}"}

    # --- Return all data ---
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": f"{probability:.2%}",
        "shap_base_value": base_value,
        "shap_values": shap_values.tolist(), 
        "feature_names": model_columns,
        "feature_values": input_df_final.iloc[0].tolist()
    }

# --- 4. Add a root endpoint ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Churn Prediction API is running."}

# --- 5. Add a main block ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)