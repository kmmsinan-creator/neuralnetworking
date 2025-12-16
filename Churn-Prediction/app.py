import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# -------------------------------------------------
# Load model & scaler (cached)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_final_model_deploy.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

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

# -------------------------------------------------
# App title & description
# -------------------------------------------------
st.title("🛒 E-Commerce Customer Churn Prediction")

st.markdown(
    """
This web application predicts whether an **e-commerce customer is likely to churn**
using a **TensorFlow Neural Network model**.

- Trained on historical customer behavior data  
- Handles class imbalance  
- Evaluated using **ROC–AUC (≈ 0.98)**  
"""
)

st.divider()

# -------------------------------------------------
# Input section
# -------------------------------------------------
st.subheader("🔢 Enter Customer Feature Values")

st.markdown(
    "Enter the customer information below. All inputs must match the same order "
    "used during model training."
)

inputs = []
for i in range(29):
    value = st.number_input(
        f"Feature {i + 1}",
        value=0.0,
        step=1.0
    )
    inputs.append(value)

st.divider()

# -------------------------------------------------
# Threshold selection
# -------------------------------------------------
st.subheader("⚙️ Decision Threshold")

threshold = st.slider(
    "Select churn decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.caption(
    "Lower threshold → catch more churners (higher recall)  \n"
    "Higher threshold → fewer false alarms (higher precision)"
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🚀 Predict Churn"):
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = model.predict(X_scaled)[0][0]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{prob:.2%}"
    )

    if prob >= threshold:
        st.error("⚠️ Customer is **likely to churn**")
    else:
        st.success("✅ Customer is **unlikely to churn**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption(
    "Model: TensorFlow Neural Network | "
    "Metric: ROC–AUC | "
    "Deployment: Streamlit Cloud"
)
