import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_final_model_deploy.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# -------------------------------------------------
# Load model & scaler
# -------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please check deployment files.")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error("❌ Scaler file not found. Please check deployment files.")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_model_and_scaler()

# -------------------------------------------------
# Feature definitions (29 features)
# -------------------------------------------------
FEATURES = [
    ("Tenure", "Number of months the customer has stayed"),
    ("PreferredLoginDevice", "Mobile = 1, Computer = 0"),
    ("CityTier", "1 = Metro, 2 = Urban, 3 = Rural"),
    ("WarehouseToHome", "Distance from warehouse to home (km)"),
    ("PreferredPaymentMode", "Encoded payment mode"),
    ("Gender", "Male = 1, Female = 0"),
    ("HourSpendOnApp", "Average hours spent on app per day"),
    ("NumberOfDeviceRegistered", "Total registered devices"),
    ("PreferedOrderCat", "Encoded preferred order category"),
    ("SatisfactionScore", "Customer satisfaction score (1–5)"),
    ("MaritalStatus", "Married = 1, Single = 0"),
    ("NumberOfAddress", "Number of saved addresses"),
    ("Complain", "Complaint raised last month (1 = Yes, 0 = No)"),
    ("OrderAmountHikeFromlastYear", "Increase in order amount (%)"),
    ("CouponUsed", "Coupons used last month"),
    ("OrderCount", "Orders placed last month"),
    ("DaySinceLastOrder", "Days since last order"),
    ("CashbackAmount", "Average cashback last month"),
    ("Feature19", "Encoded feature"),
    ("Feature20", "Encoded feature"),
    ("Feature21", "Encoded feature"),
    ("Feature22", "Encoded feature"),
    ("Feature23", "Encoded feature"),
    ("Feature24", "Encoded feature"),
    ("Feature25", "Encoded feature"),
    ("Feature26", "Encoded feature"),
    ("Feature27", "Encoded feature"),
    ("Feature28", "Encoded feature"),
    ("Feature29", "Encoded feature"),
]

# -------------------------------------------------
# Title & description
# -------------------------------------------------
st.title("🛒 E-Commerce Customer Churn Prediction")

st.markdown("""
This application predicts whether a customer is **likely to churn**
using a **TensorFlow Neural Network** trained on real e-commerce data.

**Model Performance:**  
- ROC–AUC ≈ **0.98**
""")

# -------------------------------------------------
# How to use
# -------------------------------------------------
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. Enter customer details below  
    2. Adjust the churn decision threshold  
    3. Click **Predict Churn**  
    4. View churn probability and prediction  

    All values must be numeric because categorical variables
    were encoded during training.
    """)

st.divider()

# -------------------------------------------------
# Input section
# -------------------------------------------------
st.subheader("🔢 Customer Information")

inputs = []
cols = st.columns(2)

for i, (name, desc) in enumerate(FEATURES):
    with cols[i % 2]:
        val = st.number_input(
            label=name,
            help=desc,
            value=0.0
        )
        inputs.append(val)

st.divider()

# -------------------------------------------------
# Threshold
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
    "Lower threshold → higher recall (catch more churners)\n"
    "Higher threshold → higher precision (fewer false alarms)"
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

    st.metric("Churn Probability", f"{prob:.2%}")

    if prob >= threshold:
        st.error("⚠️ Customer is **likely to churn**")
    else:
        st.success("✅ Customer is **unlikely to churn**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption(
    "Model: TensorFlow ANN | "
    "Metric: ROC–AUC | "
    "Deployment: Streamlit Cloud"
)
