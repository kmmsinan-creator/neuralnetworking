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
# LOAD MODEL & SCALER
# =====================================================
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(
        "churn_model_business.h5",
        compile=False
    )
    with open("scaler_business.pkl", "rb") as f:
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
    "NumberOfDeviceRegistered",
]

FEATURE_HELP = {
    "Tenure": "Number of months the customer has been with the company",
    "SatisfactionScore": "Customer satisfaction score (1 = very low, 5 = very high)",
    "Complain": "Whether the customer has raised any complaint",
    "DaySinceLastOrder": "Days passed since the customer's last order",
    "OrderCount": "Total number of orders placed by the customer",
    "CashbackAmount": "Total cashback received by the customer",
    "HourSpendOnApp": "Average hours spent per day on the app",
    "NumberOfDeviceRegistered": "Number of devices registered on the account",
}

# =====================================================
# TITLE & INTRO
# =====================================================
st.title("📉 Customer Churn Prediction")

st.markdown("""
This application predicts whether a customer is **likely to churn**
using **8 high-impact business features**.

### Model Highlights
- 🚀 Fast real-time inference  
- 📊 ROC-AUC ≈ **0.88**  
- 💼 Business-interpretable features  
""")

st.divider()

# =====================================================
# SINGLE CUSTOMER PREDICTION
# =====================================================
st.header("🧾 Single Customer Prediction")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=120, value=6,
        help=FEATURE_HELP["Tenure"]
    )

    satisfaction = st.slider(
        "Satisfaction Score", min_value=1, max_value=5, value=2,
        help=FEATURE_HELP["SatisfactionScore"]
    )

    complain = st.selectbox(
        "Customer Complained?", ["No", "Yes"],
        help=FEATURE_HELP["Complain"]
    )

    days_last = st.number_input(
        "Days Since Last Order", min_value=0, max_value=365, value=30,
        help=FEATURE_HELP["DaySinceLastOrder"]
    )

with col2:
    orders = st.number_input(
        "Total Order Count", min_value=0, max_value=500, value=5,
        help=FEATURE_HELP["OrderCount"]
    )

    cashback = st.number_input(
        "Cashback Amount", min_value=0.0, max_value=10000.0, value=10.0,
        help=FEATURE_HELP["CashbackAmount"]
    )

    hours = st.slider(
        "Hours Spent on App (per day)", min_value=0.0, max_value=10.0, value=1.0,
        help=FEATURE_HELP["HourSpendOnApp"]
    )

    devices = st.number_input(
        "Number of Registered Devices", min_value=1, max_value=10, value=2,
        help=FEATURE_HELP["NumberOfDeviceRegistered"]
    )

threshold = st.slider(
    "Decision Threshold",
    min_value=0.1, max_value=0.9, value=0.3,
    help="Lower threshold = catch more churners (higher recall)"
)

if st.button("🚀 Predict Churn"):
    input_data = np.array([[
        tenure,
        satisfaction,
        1 if complain == "Yes" else 0,
        days_last,
        orders,
        cashback,
        hours,
        devices
    ]])

    input_scaled = scaler.transform(input_data)
    prob = float(model.predict(input_scaled)[0][0])

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob*100:.2f}%")

    if prob >= threshold:
        st.error("❌ Customer is **LIKELY TO CHURN**")
    else:
        st.success("✅ Customer is **UNLIKELY TO CHURN**")

st.divider()

# =====================================================
# BATCH CSV PREDICTION
# =====================================================
st.header("📂 Batch CSV Prediction")

st.markdown("""
Upload a CSV file with **exactly these columns**:
Tenure, SatisfactionScore, Complain, DaySinceLastOrder,
OrderCount, CashbackAmount, HourSpendOnApp, NumberOfDeviceRegistered
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in FEATURES):
        st.error("❌ CSV file does not contain all required columns.")
    else:
        df_scaled = scaler.transform(df[FEATURES])
        df["Churn_Probability"] = model.predict(df_scaled).flatten()
        df["Churn_Prediction"] = np.where(
            df["Churn_Probability"] >= threshold,
            "Likely to Churn",
            "Unlikely to Churn"
        )

        st.success("✅ Batch prediction completed")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Final Project • Customer Churn Prediction • Streamlit + TensorFlow")

