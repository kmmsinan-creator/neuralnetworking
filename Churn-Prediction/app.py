import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os

st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# ---------------- Path Handling (CRITICAL FIX) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "churn_final_model_fixed.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------- Load Model & Scaler ----------------
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

# ---------------- UI ----------------
st.title("🛒 E-Commerce Customer Churn Prediction")
st.markdown(
    "This web app predicts whether an e-commerce customer is likely to churn "
    "using a deployed TensorFlow deep learning model."
)

st.divider()

with st.form("churn_form"):
    st.subheader("Customer Details")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    hours = st.number_input("Hours Spent on App (per day)", min_value=0.0, max_value=24.0, value=3.0)
    devices = st.number_input("Number of Registered Devices", min_value=1, max_value=10, value=2)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    cashback = st.number_input("Cashback Amount", min_value=0.0, max_value=1000.0, value=50.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    X = np.array([[tenure, hours, devices, satisfaction, cashback]])
    X_scaled = scaler.transform(X)

    prob = model.predict(X_scaled)[0][0]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob * 100:.2f}%")

    if prob >= 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

st.divider()
st.caption("TensorFlow Neural Network • Streamlit Cloud Deployment")
