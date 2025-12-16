import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(
        "churn_final_model_fixed.h5",
        compile=False
    )
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("🛒 E-Commerce Customer Churn Prediction")
st.markdown(
    "Predict customer churn using a deployed TensorFlow deep learning model."
)

st.divider()

with st.form("churn_form"):
    st.subheader("Customer Information")

    tenure = st.number_input("Tenure (months)", 0, 120, 12)
    hours = st.number_input("Hours Spent on App (daily)", 0.0, 24.0, 3.0)
    devices = st.number_input("Number of Devices", 1, 10, 2)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    cashback = st.number_input("Cashback Amount", 0.0, 1000.0, 50.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    X = np.array([[tenure, hours, devices, satisfaction, cashback]])
    X_scaled = scaler.transform(X)

    prob = model.predict(X_scaled)[0][0]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob*100:.2f}%")

    if prob >= 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

st.divider()
st.caption("TensorFlow • Streamlit • Deployed on Streamlit Cloud")
