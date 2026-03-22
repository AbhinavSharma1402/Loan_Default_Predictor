
import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
train_cols = joblib.load("columns.pkl")
train_dtypes = joblib.load("dtypes.pkl")

st.title("Loan Default Predictor")

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

    # Create input with correct schema
    input_data = pd.DataFrame(columns=train_cols)

    input_data.loc[0, 'age'] = age
    input_data.loc[0, 'income'] = income
    input_data.loc[0, 'loan_amount'] = loan_amount
    input_data.loc[0, 'Credit_Score'] = credit_score
    input_data.loc[0, 'Gender'] = gender
    input_data.loc[0, 'loan_type'] = loan_type
    input_data.loc[0, 'loan_purpose'] = loan_purpose

    # Fill missing + match types
    for col in input_data.columns:
        if train_dtypes[col] == 'object':
            input_data[col] = input_data[col].fillna('Missing').astype(str)
        else:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
            input_data[col] = input_data[col].fillna(0)

    input_data = input_data.astype(train_dtypes)

    pred = model.predict(input_data)

    if pred[0] == 1:
        st.error("Loan will Default")
    else:
        st.success("Loan will NOT Default")
