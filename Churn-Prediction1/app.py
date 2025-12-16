import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# -------------------------------
# Load Model & Scaler
# -------------------------------
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

# -------------------------------
# Title
# -------------------------------
st.title("🛒 E-Commerce Customer Churn Prediction")
st.markdown(
    """
    This application predicts **customer churn risk** using  
    **8 business-critical behavioral features** powered by a  
    **TensorFlow Neural Network**.
    """
)

st.divider()

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    Tenure = st.number_input("Tenure (months)", 0, 100, 10)
    SatisfactionScore = st.slider("Satisfaction Score", 1, 5, 3)
    OrderCount = st.number_input("Orders Last Month", 0, 50, 2)
    HourSpendOnApp = st.number_input("Hours Spent on App", 0.0, 24.0, 2.5)

with col2:
    Complain = st.selectbox("Complaint Raised?", ["No", "Yes"])
    DaySinceLastOrder = st.number_input("Days Since Last Order", 0, 365, 30)
    CashbackAmount = st.number_input("Cashback Amount", 0.0, 5000.0, 100.0)
    NumberOfDeviceRegistered = st.number_input("Devices Registered", 1, 10, 2)

Complain = 1 if Complain == "Yes" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Churn Risk"):
    input_data = np.array([[
        Tenure,
        SatisfactionScore,
        Complain,
        DaySinceLastOrder,
        OrderCount,
        CashbackAmount,
        HourSpendOnApp,
        NumberOfDeviceRegistered
    ]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{prob:.2%}"
    )

    if prob >= 0.5:
        st.error("⚠️ High Risk: Customer Likely to Churn")
        st.markdown(
            """
            **Recommended Actions:**
            - Provide targeted offers
            - Improve customer support
            - Personalized engagement
            """
        )
    else:
        st.success("✅ Low Risk: Customer Likely to Stay")
        st.markdown(
            """
            **Recommended Actions:**
            - Loyalty rewards
            - Upsell premium services
            """
        )
