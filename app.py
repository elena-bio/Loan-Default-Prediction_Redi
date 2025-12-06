# app.py
import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(page_title="Loan Default Predictor", page_icon="💰")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('best_loan_model.joblib')

try:
    model = load_model()
except:
    st.error("Model not found. Please run 'python train_model.py' first.")
    st.stop()

# UI Layout
st.title("💰 Loan Default Prediction Demo")
st.sidebar.header("User Input Features")

# Inputs
loan_amount = st.sidebar.number_input("Loan Amount", value=50000)
asset_cost = st.sidebar.number_input("Asset Cost", value=70000)
age = st.sidebar.slider("Age", 18, 80, 30)
no_of_loans = st.sidebar.number_input("Prior Loans", value=0)
no_of_curr_loans = st.sidebar.number_input("Current Loans", value=0)
education = st.sidebar.selectbox("Education", ["Graduate", "Post-Graduate", "Matriculation", "Others"])
proof_submitted = st.sidebar.selectbox("Proof", ["Aadhar", "VoterID", "PAN", "Driving", "Passport"])

# Prepare Data exactly like train_model.py
input_data = {
    'loan_amount': [loan_amount],
    'asset_cost': [asset_cost],
    'age': [age],
    'no_of_loans': [no_of_loans],
    'no_of_curr_loans': [no_of_curr_loans],
    'loan_to_asset': [loan_amount / asset_cost if asset_cost > 0 else 0], # Calculated feature
    'education': [education],
    'proof_submitted': [proof_submitted]
}

df = pd.DataFrame(input_data)

st.write("Customer Data:", df)

if st.button("Predict Risk"):
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High Risk! (Probability: {prob:.1%})")
    else:
        st.success(f" Low Risk. (Probability: {prob:.1%})")