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
# PAGE CONFIG (UI UPGRADE)
# =====================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# =====================================================
# PATH-SAFE FILE LOADING
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
# FEATURE DEFINITIONS
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
    "Tenure": "Number of months the customer has stayed with the company",
    "SatisfactionScore": "Customer satisfaction score (1 = low, 5 = high)",
    "Complain": "Whether the customer raised a complaint recently",
    "DaySinceLastOrder": "Days passed since last order",
    "OrderCount": "Number of orders placed",
    "CashbackAmount": "Cashback received last month",
    "HourSpendOnApp": "Average hours spent on the app",
    "NumberOfDeviceRegistered": "Devices registered by the customer"
}

# =====================================================
# APP HEADER
# =====================================================
st.title("📉 E-Commerce Customer Churn Prediction")

st.markdown(
    """
This **browser-based application** predicts customer churn using  
**8 business-critical features** and a **deep learning neural network**.

**Key Highlights**
- 🎯 High predictive accuracy  
- ⚡ Fast inference  
- 💼 Business interpretability  
- 📂 Supports single & bulk predictions  
"""
)

st.divider()

# =====================================================
# SINGLE CUSTOMER PREDICTION
# =====================================================
st.header("🧾 Single Customer Prediction")

left, right = st.columns([1.2, 1])

with left:
    tenure = st.number_input("Tenure (months)", 0, 120, 6, help=FEATURE_HELP["Tenure"])
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3, help=FEATURE_HELP["SatisfactionScore"])
    complain = st.selectbox("Customer Complained?", ["No", "Yes"], help=FEATURE_HELP["Complain"])
    days_last_order = st.number_input("Days Since Last Order", 0, 365, 60, help=FEATURE_HELP["DaySinceLastOrder"])
    order_count = st.number_input("Total Order Count", 0, 1000, 5, help=FEATURE_HELP["OrderCount"])
    cashback = st.number_input("Cashback Amount", 0.0, 1000.0, 10.0, help=FEATURE_HELP["CashbackAmount"])
    hours_app = st.slider("Hours Spent on App", 0.0, 10.0, 1.0, help=FEATURE_HELP["HourSpendOnApp"])
    devices = st.number_input("Registered Devices", 1, 10, 2, help=FEATURE_HELP["NumberOfDeviceRegistered"])

complain = 1 if complain == "Yes" else 0

with right:
    st.subheader("⚙️ Decision Threshold")
    threshold = st.slider(
        "Churn Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05
    )

    st.caption("Lower threshold → higher recall")
    st.caption("Higher threshold → higher precision")

# =====================================================
# PREDICTION BUTTON
# =====================================================
if st.button("🚀 Predict Churn", use_container_width=True):
    input_data = np.array([[tenure, satisfaction, complain,
                            days_last_order, order_count,
                            cashback, hours_app, devices]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{prob*100:.2f}%")
    st.progress(min(prob, 1.0))

    if prob >= threshold:
        st.error("❌ Customer is LIKELY to churn")
    else:
        st.success("✅ Customer is UNLIKELY to churn")

# =====================================================
# BULK PREDICTION – RAW CSV
# =====================================================
st.divider()
st.header("📂 Bulk Prediction – Raw Customer Data")

st.markdown(
    """
Upload a CSV file with **RAW values** containing these columns:

`Tenure, SatisfactionScore, Complain, DaySinceLastOrder,
OrderCount, CashbackAmount, HourSpendOnApp, NumberOfDeviceRegistered`
"""
)

raw_file = st.file_uploader("Upload RAW customer CSV", type=["csv"], key="raw")

if raw_file is not None:
    df_raw = pd.read_csv(raw_file)

    if not all(col in df_raw.columns for col in FEATURES):
        st.error("❌ CSV columns do not match required feature list.")
    else:
        X_scaled = scaler.transform(df_raw[FEATURES])
        probs = model.predict(X_scaled).flatten()

        df_raw["Churn_Probability"] = probs
        df_raw["Churn_Prediction"] = (probs >= threshold).astype(int)

        st.success("✅ Bulk prediction completed (RAW data)")
        st.dataframe(df_raw, use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions",
            df_raw.to_csv(index=False),
            "bulk_churn_predictions_raw.csv",
            "text/csv"
        )

# =====================================================
# BULK PREDICTION – PRE-SCALED CSV (NEW)
# =====================================================
st.divider()
st.header("📂 Bulk Prediction – Pre-Scaled CSV")

st.markdown(
    """
Upload a **PRE-SCALED CSV** (already scaled using the same scaler).
This option is useful for **offline preprocessing pipelines**.
"""
)

scaled_file = st.file_uploader("Upload SCALED CSV", type=["csv"], key="scaled")

if scaled_file is not None:
    df_scaled = pd.read_csv(scaled_file)

    probs = model.predict(df_scaled.values).flatten()

    df_scaled["Churn_Probability"] = probs
    df_scaled["Churn_Prediction"] = (probs >= threshold).astype(int)

    st.success("✅ Bulk prediction completed (SCALED data)")
    st.dataframe(df_scaled, use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions",
        df_scaled.to_csv(index=False),
        "bulk_churn_predictions_scaled.csv",
        "text/csv"
    )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("© E-Commerce Customer Churn Prediction | Neural Network | Streamlit Cloud")
