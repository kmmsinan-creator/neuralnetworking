# =====================================================
# CUSTOMER CHURN PREDICTION – FINAL STABLE STREAMLIT APP
# =====================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model_business.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_business.pkl")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------------- FEATURES ----------------
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

# ---------------- HEADER ----------------
st.title("📉 E-Commerce Customer Churn Prediction")
st.markdown(
    """
Predict customer churn using a **deep learning model** trained on  
**business-critical behavioral features**.
"""
)

st.divider()

# =====================================================
# SINGLE CUSTOMER PREDICTION (CLEAN CARD)
# =====================================================
st.subheader("🧍 Single Customer Prediction")

with st.container(border=True):
    tenure = st.number_input("Tenure (months)", 0, 120, 6)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    complain = st.radio("Complaint Raised?", ["No", "Yes"], horizontal=True)
    days_last = st.number_input("Days Since Last Order", 0, 365, 60)
    order_count = st.number_input("Order Count", 0, 1000, 5)
    cashback = st.number_input("Cashback Amount", 0.0, 1000.0, 10.0)
    hours_app = st.slider("Hours Spent on App", 0.0, 10.0, 1.0)
    devices = st.number_input("Devices Registered", 1, 10, 2)

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.3, step=0.05)

    if st.button("🚀 Predict", use_container_width=True):
        complain_val = 1 if complain == "Yes" else 0

        X = np.array([[tenure, satisfaction, complain_val,
                       days_last, order_count,
                       cashback, hours_app, devices]])

        X_scaled = scaler.transform(X)
        prob = float(model.predict(X_scaled)[0][0])

        st.metric("Churn Probability", f"{prob*100:.2f}%")

        if prob >= threshold:
            st.error("❌ High risk of churn")
        else:
            st.success("✅ Low risk of churn")

st.divider()

# =====================================================
# BULK CSV PREDICTION (RAW ONLY – SAFE)
# =====================================================
st.subheader("📂 Bulk Customer Prediction")

st.markdown(
    """
Upload a CSV file with **these exact columns**:

`Tenure, SatisfactionScore, Complain, DaySinceLastOrder,  
OrderCount, CashbackAmount, HourSpendOnApp, NumberOfDeviceRegistered`
"""
)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    if not all(col in df.columns for col in FEATURES):
        st.error("❌ CSV does not contain required columns.")
    else:
        X_scaled = scaler.transform(df[FEATURES])
        probs = model.predict(X_scaled).flatten()

        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = (probs >= threshold).astype(int)

        st.success("✅ Bulk prediction completed")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "churn_predictions.csv",
            "text/csv"
        )

# ---------------- FOOTER ----------------
st.divider()
st.caption("Neural Network Churn Prediction | Streamlit Cloud")
