import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# =====================================================
# Resolve file paths safely (Streamlit Cloud safe)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "churn_model_business.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_business.pkl")

# =====================================================
# Load model & scaler (cached)
# =====================================================
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# =====================================================
# App Title & Description
# =====================================================
st.title("📉 Customer Churn Prediction")
st.markdown(
    """
This application predicts whether a customer is likely to **churn**
based on **8 high-impact business features**.

The model is optimized for:
- High predictive performance  
- Fast real-time inference  
- Business interpretability  
"""
)

st.divider()

# =====================================================
# Feature Inputs (8 Business Features)
# =====================================================
st.subheader("🧾 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    complain = st.selectbox("Customer Complained?", ["No", "Yes"])
    days_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)

with col2:
    order_count = st.number_input("Total Order Count", min_value=0, max_value=500, value=20)
    cashback = st.number_input("Cashback Amount", min_value=0.0, value=50.0)
    hours_app = st.slider("Hours Spent on App", 0.0, 10.0, 2.5)
    devices = st.selectbox("Number of Registered Devices", [1, 2, 3, 4])

# Convert categorical to numeric
complain_val = 1 if complain == "Yes" else 0

# =====================================================
# Prediction Threshold
# =====================================================
st.divider()
st.subheader("⚙️ Decision Threshold")

threshold = st.slider(
    "Select churn decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.caption(
    "Lower threshold → higher recall (catch more churners)\n\n"
    "Higher threshold → higher precision (fewer false alarms)"
)

# =====================================================
# Predict Button
# =====================================================
st.divider()

if st.button("🚀 Predict Churn", use_container_width=True):

    # Prepare input
    input_data = np.array([[
        tenure,
        satisfaction,
        complain_val,
        days_last_order,
        order_count,
        cashback,
        hours_app,
        devices
    ]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prob = model.predict(input_scaled)[0][0]
    prediction = int(prob >= threshold)

    # =================================================
    # Results
    # =================================================
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{prob * 100:.2f}%"
    )

    if prediction == 1:
        st.error("❌ Customer is **LIKELY to churn**")
    else:
        st.success("✅ Customer is **UNLIKELY to churn**")

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption(
    "Model trained using 8 business-critical features | "
    "Deployed with Streamlit Cloud | TensorFlow"
)
