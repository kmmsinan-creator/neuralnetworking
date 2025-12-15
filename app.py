import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# Load model & scaler
model = load_model("churn_final_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🛒 E-Commerce Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to **churn** using a deep learning model."
)

st.divider()

# -------- Input Form --------
with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    hours = st.number_input("Hours Spent on App", 0, 24, 3)
    devices = st.number_input("Number of Devices Registered", 1, 10, 2)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    cashback = st.number_input("Cashback Amount", 0.0, 500.0, 50.0)

    submitted = st.form_submit_button("Predict Churn")

# -------- Prediction --------
if submitted:
    input_data = np.array([[tenure, hours, devices, satisfaction, cashback]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
