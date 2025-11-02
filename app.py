# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="CodeAlpha Credit Scoring", layout="centered")

st.title("💳 CodeAlpha — Credit Scoring Prediction")
st.write("Enter financial details and predict creditworthiness.")

# Load artifacts
ARTIFACT_DIR = 'artifacts'
if not os.path.exists(ARTIFACT_DIR):
    st.error("No artifacts found. Run credit_scoring.py first to train and save models.")
    st.stop()

# load scaler and model
scaler = joblib.load(f'{ARTIFACT_DIR}/scaler.pkl')
model = joblib.load(f'{ARTIFACT_DIR}/model.pkl')

# UI inputs (sidebar)
st.sidebar.header("Input Features")
income = st.sidebar.number_input("Annual Income (₹)", min_value=0, value=50000, step=1000)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
debt = st.sidebar.number_input("Total Debt (₹)", min_value=0, value=5000, step=500)
payment_history = st.sidebar.selectbox("Payment History", ["Good (1)", "Bad (0)"])
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, value=15000, step=500)

payment_history_val = 1 if payment_history.startswith("Good") else 0

import pandas as pd  # make sure you have this import at top

# Make DataFrame with same column names as training
sample_df = pd.DataFrame([[income, age, debt, payment_history_val, loan_amount]],
                         columns=['income', 'age', 'debt', 'payment_history', 'loan_amount'])

sample_scaled = scaler.transform(sample_df)

if st.button("Predict Creditworthiness"):
    pred = model.predict(sample_scaled)[0]
    # try probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_scaled)[0][1]
        proba_text = f"{proba*100:.1f}%"
    else:
        proba_text = "N/A"

    if pred == 1:
        st.success(f"✅ Predicted: Creditworthy (probability of being creditworthy: {proba_text})")
    else:
        st.error(f"⚠️ Predicted: Not Creditworthy (probability of being creditworthy: {proba_text})")

st.markdown("---")
st.subheader("Model comparison (saved metrics)")
metrics_path = f'{ARTIFACT_DIR}/metrics.csv'
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)
    st.dataframe(df.style.format({
        'accuracy': "{:.3f}",
        'precision': "{:.3f}",
        'recall': "{:.3f}",
        'f1_score': "{:.3f}",
        'roc_auc': "{:.3f}"
    }))
else:
    st.write("No metrics file found. Run training script to generate metrics.")
