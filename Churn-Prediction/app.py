import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(
        "churn_final_model.keras",
        compile=False   # IMPORTANT: avoids optimizer deserialization issues
    )
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_model_and_scaler()

# ---------------- UI ----------------
st.title("🛒 E-Commerce Customer Churn Prediction")
st.markdown(
    """
    This application predicts whether a customer is likely to **churn**
    using a **deep learning model (TensorFlow)** trained on e-commerce data.
    """
)

st.divider()

# ---------------- Input Form ----------------
with st.form("churn_form"):
    st.subheader("📥 Enter Customer Details")

    tenure = st.number_input(
        "Tenure (months with company)",
        min_value=0,
        max_value=120,
        value=12
    )

    hours = st.number_input(
        "Hours Spent on App (per day)",
        min_value=0.0,
        max_value=24.0,
        value=3.0
    )

    devices = st.number_input(
        "Number of Devices Registered",
        min_value=1,
        max_value=10,
        value=2
    )

    satisfaction = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    cashback = st.number_input(
        "Cashback Amount (last month)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0
    )

    submit = st.form_submit_button("🔮 Predict Churn")

# ---------------- Prediction ----------------
if submit:
    input_data = np.array([
        [tenure, hours, devices, satisfaction, cashback]
    ])

    input_scaled = scaler.transform(input_data)
    churn_prob = model.predict(input_scaled)[0][0]

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{churn_prob * 100:.2f}%"
    )

    if churn_prob >= 0.5:
        st.error("⚠️ Customer is **LIKELY TO CHURN**")
    else:
        st.success("✅ Customer is **LIKELY TO STAY**")

# ---------------- Footer ----------------
st.divider()
st.caption(
    "Developed using TensorFlow & Streamlit | "
    "Deployed on Streamlit Community Cloud"
)
