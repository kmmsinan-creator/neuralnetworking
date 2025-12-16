# =====================================================
# CUSTOMER CHURN PREDICTION – FINAL STREAMLIT APP
# =====================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# =====================================================
# PATH-SAFE FILE LOADING (CRITICAL FIX)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "churn_model_business.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_business.pkl")

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# =====================================================
# FEATURE DEFINITIONS (BUSINESS FEATURES)
# =====================================================
FEATURES = [
    "Tenure",
    "SatisfactionScore",
    "Complain",
    "DaySinceLastOrder",
    "OrderCount",
    "CashbackAmount",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered"
]

FEATURE_HELP = {
    "Tenure": "Number of months the customer has been with the company",
    "SatisfactionScore": "Customer satisfaction rating (1 = very low, 5 = very high)",
    "Complain": "Whether the customer has raised complaints (1 = Yes, 0 = No)",
    "DaySinceLastOrder": "Days since the customer's last order",
    "OrderCount": "Total number of orders placed",
    "CashbackAmount": "Total cashback received",
    "HourSpendOnApp": "Average hours spent on the app per day",
    "NumberOfDeviceRegistered": "Number of devices registered by the customer"
}

# =====================================================
# APP HEADER
# =====================================================
st.title("📉 Customer Churn Prediction")

st.markdown(
    """
This application predicts whether a customer is **likely to churn**
using **8 high-impact business features**.

**Optimized for:**
- 🎯 Strong predictive performance  
- ⚡ Fast real-time inference  
- 💼 Business interpretability  
"""
)

st.divider()

# =====================================================
# SINGLE CUSTOMER INPUT
# =====================================================
st.subheader("🧾 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 120, 6, help=FEATURE_HELP["Tenure"])
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3, help=FEATURE_HELP["SatisfactionScore"])
    cashback = st.number_input("Cashback Amount", 0.0, 1000.0, 10.0, help=FEATURE_HELP["CashbackAmount"])
    days_last_order = st.number_input("Days Since Last Order", 0, 365, 60, help=FEATURE_HELP["DaySinceLastOrder"])

with col2:
    order_count = st.number_input("Total Order Count", 0, 1000, 5, help=FEATURE_HELP["OrderCount"])
    hours_app = st.slider("Hours Spent on App", 0.0, 10.0, 1.0, help=FEATURE_HELP["HourSpendOnApp"])
    devices = st.number_input("Registered Devices", 1, 10, 2, help=FEATURE_HELP["NumberOfDeviceRegistered"])
    complain = st.selectbox("Customer Complained?", ["No", "Yes"], help=FEATURE_HELP["Complain"])

complain = 1 if complain == "Yes" else 0

# =====================================================
# DECISION THRESHOLD
# =====================================================
st.divider()
st.subheader("⚙️ Decision Threshold")

threshold = st.slider(
    "Select churn decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05
)

st.caption("Lower threshold → catch more churners (higher recall)")
st.caption("Higher threshold → fewer false alarms (higher precision)")

# =====================================================
# PREDICTION
# =====================================================
st.divider()

if st.button("🚀 Predict Churn", use_container_width=True):

    input_data = np.array([[
        tenure,
        satisfaction,
        complain,
        days_last_order,
        order_count,
        cashback,
        hours_app,
        devices
    ]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob * 100:.2f}%")

    if prob >= threshold:
        st.error("❌ Customer is LIKELY to churn")
    else:
        st.success("✅ Customer is UNLIKELY to churn")

# =====================================================
# BATCH CSV PREDICTION
# =====================================================
st.divider()
st.subheader("📂 Batch Prediction (CSV Upload)")

st.markdown(
    """
Upload a CSV file with **exactly these columns**:
Tenure, SatisfactionScore, Complain, DaySinceLastOrder,
OrderCount, CashbackAmount, HourSpendOnApp, NumberOfDeviceRegistered"""
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in FEATURES):
        st.error("❌ CSV columns do not match required feature list.")
    else:
        df_scaled = scaler.transform(df[FEATURES])
        df["Churn_Probability"] = model.predict(df_scaled)
        df["Churn_Prediction"] = (df["Churn_Probability"] >= threshold).astype(int)

        st.success("✅ Batch prediction completed")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Predictions",
            df.to_csv(index=False),
            "churn_predictions.csv",
            "text/csv"
        )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("© Customer Churn Prediction | Neural Network Model | Streamlit App")
