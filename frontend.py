import streamlit as st
import requests
import json
import pandas as pd
import shap
import numpy as np
import streamlit.components.v1 as components

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
API_URL = "http://3.228.187.148/predict"
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ----------------------------------------------------
# Helper Function
# ----------------------------------------------------
def get_prediction(data):
    """
    Sends data to the FastAPI endpoint and gets a prediction.
    """
    try:
        response = requests.post(API_URL, data=json.dumps(data))
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "ConnectionError: Could not connect to the API. Is the server running?"}
    except Exception as e:
        st.error(f"API Error: {e}")
        try:
            st.error(f"Response content: {response.text}")
        except NameError:
            pass
        return {"error": f"An error occurred: {e}"}

def process_input(data):
    """
    Converts human-readable frontend inputs into the one-hot encoded
    format that the prediction API expects (using underscore names).
    """
    
    # This dictionary must match the Pydantic model in your 'app.py'
    api_data = {
        "tenure": data['tenure'],
        "MonthlyCharges": data['MonthlyCharges'],
        "TotalCharges": data['TotalCharges'],
        "SeniorCitizen": 1 if data['SeniorCitizen'] == 'Yes' else 0,
        "gender_Male": 1 if data['gender'] == 'Male' else 0,
        "Partner_Yes": 1 if data['Partner'] == 'Yes' else 0,
        "Dependents_Yes": 1 if data['Dependents'] == 'Yes' else 0,
        "PhoneService_Yes": 1 if data['PhoneService'] == 'Yes' else 0,
        "MultipleLines_No_phone_service": 1 if data['MultipleLines'] == 'No phone service' else 0,
        "MultipleLines_Yes": 1 if data['MultipleLines'] == 'Yes' else 0,
        "InternetService_Fiber_optic": 1 if data['InternetService'] == 'Fiber optic' else 0,
        "InternetService_No": 1 if data['InternetService'] == 'No' else 0,
        "OnlineSecurity_No_internet_service": 1 if data['OnlineSecurity'] == 'No internet service' else 0,
        "OnlineSecurity_Yes": 1 if data['OnlineSecurity'] == 'Yes' else 0,
        "OnlineBackup_No_internet_service": 1 if data['OnlineBackup'] == 'No internet service' else 0,
        "OnlineBackup_Yes": 1 if data['OnlineBackup'] == 'Yes' else 0,
        "DeviceProtection_No_internet_service": 1 if data['DeviceProtection'] == 'No internet service' else 0,
        "DeviceProtection_Yes": 1 if data['DeviceProtection'] == 'Yes' else 0,
        "TechSupport_No_internet_service": 1 if data['TechSupport'] == 'No internet service' else 0,
        "TechSupport_Yes": 1 if data['TechSupport'] == 'Yes' else 0,
        "StreamingTV_No_internet_service": 1 if data['StreamingTV'] == 'No internet service' else 0,
        "StreamingTV_Yes": 1 if data['StreamingTV'] == 'Yes' else 0,
        "StreamingMovies_No_internet_service": 1 if data['StreamingMovies'] == 'No internet service' else 0,
        "StreamingMovies_Yes": 1 if data['StreamingMovies'] == 'Yes' else 0,
        "Contract_One_year": 1 if data['Contract'] == 'One year' else 0,
        "Contract_Two_year": 1 if data['Contract'] == 'Two year' else 0,
        "PaperlessBilling_Yes": 1 if data['PaperlessBilling'] == 'Yes' else 0,
        "PaymentMethod_Credit_card_automatic": 1 if data['PaymentMethod'] == 'Credit card (automatic)' else 0,
        "PaymentMethod_Electronic_check": 1 if data['PaymentMethod'] == 'Electronic check' else 0,
        "PaymentMethod_Mailed_check": 1 if data['PaymentMethod'] == 'Mailed check' else 0
    }
    
    return api_data

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("Customer Churn Prediction 🔮")
st.markdown("Enter the customer's details to predict whether they will churn.")

# Use a dictionary to hold all inputs
inputs = {}

# --- Create columns for a cleaner layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Customer Info")
    inputs['gender'] = st.radio("Gender", ["Female", "Male"])
    inputs['Partner'] = st.radio("Has Partner?", ["No", "Yes"])
    inputs['Dependents'] = st.radio("Has Dependents?", ["No", "Yes"])
    inputs['SeniorCitizen'] = st.radio("Senior Citizen?", ["No", "Yes"])

with col2:
    st.header("Account Details")
    inputs['tenure'] = st.slider("Tenure (months)", 0, 72, 12)
    inputs['MonthlyCharges'] = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    inputs['TotalCharges'] = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)

with col3:
    st.header("Contract & Billing")
    inputs['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    inputs['PaymentMethod'] = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    inputs['PaperlessBilling'] = st.radio("Paperless Billing?", ["No", "Yes"])

st.divider()
st.header("Services")
col4, col5, col6 = st.columns(3)

with col4:
    inputs['PhoneService'] = st.selectbox("Phone Service?", ["No", "Yes"])
    inputs['MultipleLines'] = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
    inputs['InternetService'] = st.selectbox("Internet Service?", ["DSL", "Fiber optic", "No"])

with col5:
    inputs['OnlineSecurity'] = st.selectbox("Online Security?", ["No", "Yes", "No internet service"])
    inputs['OnlineBackup'] = st.selectbox("Online Backup?", ["No", "Yes", "No internet service"])
    inputs['DeviceProtection'] = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])

with col6:
    inputs['TechSupport'] = st.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
    inputs['StreamingTV'] = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    inputs['StreamingMovies'] = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])

st.divider()

# --- Prediction Button ---
if st.button("Predict Churn", type="primary"):
    with st.spinner("Processing..."):
        # 1. Process inputs
        api_data = process_input(inputs)
        
        # 2. Get prediction
        result = get_prediction(api_data)
        
        # 3. Display result
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            prediction = result.get("prediction")
            probability = result.get("churn_probability", "N/A")
            
            if prediction == "Churn":
                st.error(f"Prediction: **{prediction}**")
                st.warning(f"Probability of Churn: **{probability}**")
            else:
                st.success(f"Prediction: **{prediction}**")
                st.info(f"Probability of Churn: **{probability}**")
            
            # --- TEXT-BASED SHAP EXPLANATION ---
            st.divider()
            st.subheader("Why this prediction? (Explainable AI)")
            
            try:
                # Load the data from the API response
                shap_values = np.array(result.get("shap_values"))
                feature_names = result.get("feature_names")
                feature_values = np.array(result.get("feature_values"))

                # Combine features and their SHAP values
                impacts = []
                for i in range(len(feature_names)):
                    impacts.append((feature_names[i], feature_values[i], shap_values[i]))

                # A positive SHAP value pushes towards CHURN (Class 1)
                # A negative SHAP value pushes towards NO CHURN (Class 0)
                
                # Sort by the magnitude of the SHAP value
                pushing_features = sorted([f for f in impacts if f[2] > 0], key=lambda x: x[2], reverse=True)
                pulling_features = sorted([f for f in impacts if f[2] < 0], key=lambda x: x[2])

                st.markdown("This prediction is based on the following key factors:")

                if prediction == "Churn":
                    st.write("#### Top factors pushing towards **Churn** (positive impact):")
                    for feature, value, impact in pushing_features[:3]:
                        st.markdown(f"- **{feature}** (Value: `{value}`)")
                else:
                    st.write("#### Top factors pushing towards **No Churn** (negative impact):")
                    for feature, value, impact in pulling_features[:3]:
                        st.markdown(f"- **{feature}** (Value: `{value}`)")

            except Exception as e:
                st.error(f"Could not render text explanation: {e}")
            # --- END OF TEXT-BASED SHAP SECTION ---