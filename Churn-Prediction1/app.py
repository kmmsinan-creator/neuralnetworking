import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# ------------------------------------
# Page config
# ------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ------------------------------------
# Load model & scaler
# ------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(
        "churn_model_business_8.h5",
        compile=False
    )
    with open("scaler_business_8.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ------------------------------------
# Title & description
# ------------------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown(
    """
This application predicts whether a customer is **likely to churn**  
based on **8 key business-focused behavioral features**.

👉 Designed for **business decision-making**  
👉 Lightweight, fast & interpretable
"""
)

st.divider()

# ------------------------------------
# Input form
# ------------------------------------
st.subheader("🔢 Enter Customer Details")

with st.form("churn_form"):
    Tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    SatisfactionScore = st.slider("Satisfaction Score (1 = Low, 5 = High)", 1, 5, 3)
    Complain = st.selectbox("Has the customer complained?", ["No", "Yes"])
    DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
    OrderCount = st.number_input("Total Order Count", min_value=0, max_value=500, value=10)
    CashbackAmount = st.number_input("Cashback Amount", min_value=0.0, max_value=5000.0, value=100.0)
    HourSpendOnApp = st.number_input("Hours Spent on App (per month)", min_value=0.0, max_value=500.0, value=20.0)
    NumberOfDeviceRegistered = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=2)

    submit = st.form_submit_button("🔍 Predict Churn")

# ------------------------------------
# Prediction
# ------------------------------------
if submit:
    complain_value = 1 if Complain == "Yes" else 0

    input_data = np.array([[
        Tenure,
        SatisfactionScore,
        complain_value,
        DaySinceLastOrder,
        OrderCount,
        CashbackAmount,
        HourSpendOnApp,
        NumberOfDeviceRegistered
    ]])

    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled)[0][0]

    st.divider()
    st.subheader("📈 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if probability >= 0.5:
        st.error("⚠️ High Risk: Customer is likely to churn")
    else:
        st.success("✅ Low Risk: Customer is likely to stay")

    st.caption(
        "Prediction is based on a neural network trained using business-critical features."
    )

# ------------------------------------
# Footer
# ------------------------------------
st.divider()
st.markdown(
    """
**Model:** Neural Network (TensorFlow)  
**Features:** 8 business-selected inputs  
**Metric:** ROC–AUC optimized  

Developed for academic & business demonstration.
"""
)
