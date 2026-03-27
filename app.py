import streamlit as st
import pandas as pd
import joblib
import os

# Safe path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load files
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
train_cols = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

st.title("Loan Default Predictor")

st.info("This app uses key features. Remaining features are auto-filled.")

st.header("Enter Loan Details")

# Inputs
age = st.number_input("Age", value=30)
income = st.number_input("Income", value=100000)
loan_amount = st.number_input("Loan Amount", value=200000)
credit_score = st.number_input("Credit Score", value=750)

gender = st.selectbox("Gender", ['Male', 'Female', 'Joint', 'Sex Not Available'])
loan_type = st.selectbox("Loan Type", ['type1', 'type2', 'type3'])
loan_purpose = st.selectbox("Loan Purpose", ['p1', 'p2', 'p3', 'p4'])

if st.button("Predict"):

    try:
        # Create empty input
        input_data = pd.DataFrame(columns=train_cols)

        # Fill all with None (important)
        input_data.loc[0] = None

        # Fill user inputs
        input_data.loc[0, 'age'] = age
        input_data.loc[0, 'income'] = income
        input_data.loc[0, 'loan_amount'] = loan_amount
        input_data.loc[0, 'Credit_Score'] = credit_score
        input_data.loc[0, 'Gender'] = gender
        input_data.loc[0, 'loan_type'] = loan_type
        input_data.loc[0, 'loan_purpose'] = loan_purpose

        # Predict
        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]

        # Output
        if pred[0] == 1:
            st.error(f"Loan will Default ❌ (Prob: {prob:.2f})")
        else:
            st.success(f"Loan will NOT Default ✅ (Prob: {prob:.2f})")

    except Exception as e:
        st.error("Error occurred:")
        st.write(str(e))
