# Loan_Default_Predictor
To predict whether a person will default a loan or not based on some info

#  Loan Default Prediction System

An end-to-end Machine Learning project that predicts whether a loan applicant is likely to default, built using real-world financial data and deployed as an interactive web application.

---

##  Overview

This project focuses on building a **robust and reliable loan default prediction system** using modern ML practices:

* Cleaned and analyzed real-world dataset
* Built a complete ML pipeline with preprocessing
* Identified and removed **data leakage features**
* Compared multiple models (Random Forest vs XGBoost)
* Deployed as a **Streamlit web app**

---

##  Problem Statement

Banks need to assess whether a customer will default on a loan.

 Goal:
Predict **loan default (Yes / No)** based on customer and financial data.

---

##  Key Features

*  End-to-end ML pipeline using `Scikit-learn`
*  Data preprocessing with `ColumnTransformer`
*  Feature engineering and EDA
*  Removed leakage features (e.g., interest rate, DTI)
*  Model comparison:

  * Random Forest
  * XGBoost (Best performer)
*  Achieved **~88.5% accuracy**
*  Fully deployed using Streamlit

---

##  Important Insight (Data Leakage Fix)

Initially, the model achieved **100% accuracy**, which was unrealistic.

 Root cause:

* Features like `rate_of_interest`, `dtir1`, etc. were derived from risk assessment

 Solution:

* Removed these **post-decision features**
* Retrained model → realistic performance

 This demonstrates strong understanding of **real-world ML pitfalls**

---

##  Model Performance

| Model         | Accuracy          |
| ------------- | ----------------- |
| Random Forest | ~87%              |
| XGBoost       | **~88.7% (Best)** |

---

##  Tech Stack

* Python 
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

##  Project Structure

```
loan-default-predictor/
│
├── app.py              # Streamlit app
├── model.pkl           # Trained ML model
├── columns.pkl         # Feature schema
├── dtypes.pkl          # Data types for inference
├── requirements.txt
├── README.md
```

---

##  How to Run Locally

```bash
git clone https://github.com/yourusername/loan-default-predictor.git
cd loan-default-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

##  Live Demo

 [Add your Streamlit link here]

---

##  Learnings & Highlights

* Importance of **feature selection**
* Understanding and fixing **data leakage**
* Building **reproducible ML pipelines**
* Handling **real-world deployment issues**
* Aligning training and inference schema

---

##  Future Improvements

* Add full input UI for all features
* Improve model with hyperparameter tuning
* Add probability/confidence output
* Deploy using Docker / cloud services

---

##  Connect

If you found this project useful or interesting, feel free to connect!

* GitHub: [Your Profile]


Give it a ⭐ — it helps a lot!

