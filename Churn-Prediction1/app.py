# =====================================================
# E-COMMERCE CUSTOMER CHURN PREDICTION (FINAL VERSION)
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
    page_title="E-Commerce Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# =====================================================
# SAFE FILE PATHS
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
# FEATURE DEFINITIONS (8 BUSINESS FEATURES)
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
    "Complain": "Whether the customer has raised a complaint",
    "DaySinceLastOrder": "Days since the customer last placed an order",
    "OrderCount": "Total number of orders placed by the customer",
    "CashbackAmount": "Total cashback received by the customer",
    "HourSpendOnApp": "Average daily time spent on the app (hours)",
    "NumberOfDeviceRegistered": "Number of devices registered by the customer"
}

# =====================================================
# APP HEADER
# =====================================================
st.title("📉 E-Commerce Customer Churn Prediction")

st.markdown("""
Predict customer churn using a **Feedforward Neural Network (MLP)** trained on  
**business-critical behavioral features**.

**Why this app?**
- 🎯 Interpretable business features  
- ⚡ Fast real-time predictions  
- 📊 Risk-based decision support  
""")

st.divider()

# =====================================================
# SINGLE CUSTOMER INPUT
# =====================================================
st.subheader("👤 Single Customer Prediction")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 120, 6, help=FEATURE_HELP["Tenure"])
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3, help=FEATURE_HELP["SatisfactionScore"])
    cashback = st.number_input("Cashback Amount", 0.0, 2000.0, 50.0, help=FEATURE_HELP["CashbackAmount"])
    days_last_order = st.number_input("Days Since Last Order", 0, 365, 30, help=FEATURE_HELP["DaySinceLastOrder"])

with col2:
    order_count = st.number_input("Total Orders", 0, 1000, 10, help=FEATURE_HELP["OrderCount"])
    hours_app = st.slider("Hours Spent on App", 0.0, 10.0, 1.5, help=FEATURE_HELP["HourSpendOnApp"])
    devices = st.number_input("Registered Devices", 1, 10, 2, help=FEATURE_HELP["NumberOfDeviceRegistered"])
    complain = st.radio("Complaint Raised?", ["No", "Yes"], help=FEATURE_HELP["Complain"])

complain = 1 if complain == "Yes" else 0

# =====================================================
# DECISION THRESHOLD
# =====================================================
st.divider()
st.subheader("⚙️ Decision Threshold")

threshold = st.slider(
    "Churn decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05
)

st.caption("Lower threshold → catch more churners (higher recall)")
st.caption("Higher threshold → fewer false alarms (higher precision)")

# =====================================================
# SINGLE PREDICTION
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
    prob = float(model.predict(input_scaled)[0][0])

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob * 100:.2f}%")

    # ✅ BUSINESS-CORRECT RISK LOGIC
    if prob < 0.25:
        st.success("🟢 Low churn risk — customer likely to stay")
    elif prob < 0.50:
        st.warning("🟡 Medium churn risk — monitor customer")
    else:
        st.error("🔴 High churn risk — retention action recommended")

# =====================================================
# BULK CSV PREDICTION
# =====================================================
st.divider()
st.subheader("📂 Bulk Customer Prediction (CSV Upload)")

st.markdown("""
Upload a CSV file containing **raw (unscaled) values** with **exactly these columns**:

`Tenure, SatisfactionScore, Complain, DaySinceLastOrder,  
OrderCount, CashbackAmount, HourSpendOnApp, NumberOfDeviceRegistered`
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 🔒 Column validation
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        df_scaled = scaler.transform(df[FEATURES])
        probs = model.predict(df_scaled).flatten()

        df["Churn_Probability"] = probs

        # Risk buckets
        df["Churn_Risk"] = pd.cut(
            probs,
            bins=[0, 0.25, 0.5, 1.0],
            labels=["Low", "Medium", "High"]
        )

        st.success("✅ Bulk prediction completed")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Prediction Results",
            data=df.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("© 2025 | Customer Churn Prediction | Neural Network (TensorFlow + Streamlit)")
