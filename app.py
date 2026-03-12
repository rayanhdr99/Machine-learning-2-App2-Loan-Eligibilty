"""
app.py - Streamlit application for Loan Eligibility Prediction.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.preprocessor import impute_missing, encode_and_prepare, split_and_scale
from src.model import train_logistic_regression, train_decision_tree, train_random_forest
from src.evaluator import evaluate_model, cross_validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "credit.csv")

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="💰", layout="wide")


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_resource
def get_models():
    df = get_data()
    df = impute_missing(df)
    df_encoded = encode_and_prepare(df)
    X_train_s, X_test_s, y_train, y_test, scaler, feature_cols = split_and_scale(df_encoded)
    lr = train_logistic_regression(X_train_s, y_train)
    dt = train_decision_tree(X_train_s, y_train)
    rf = train_random_forest(X_train_s, y_train)
    return lr, dt, rf, X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, df_encoded


def main():
    st.title("💰 Loan Eligibility Predictor")
    st.markdown("Predict whether a loan application should be **approved** or **denied** using classification models.")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "Model Performance", "Predict Eligibility"])

    try:
        df_raw = get_data()
        lr, dt, rf, X_train_s, X_test_s, y_train, y_test, scaler, feature_cols, df_enc = get_models()
    except Exception as e:
        st.error(f"Error during setup: {e}")
        logger.error("App startup error: %s", e)
        return

    if page == "Dataset Overview":
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Applicants", df_raw.shape[0])
        approved = (df_raw["Loan_Approved"] == "Y").sum()
        col2.metric("Approved", f"{approved} ({approved/len(df_raw)*100:.0f}%)")
        col3.metric("Denied", f"{len(df_raw)-approved} ({(len(df_raw)-approved)/len(df_raw)*100:.0f}%)")

        st.subheader("Sample Data")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Loan Approval Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df_raw["Loan_Approved"].value_counts().plot.bar(ax=ax, color=["steelblue", "salmon"], edgecolor="white")
        ax.set_xticklabels(["Approved (Y)", "Denied (N)"], rotation=0)
        ax.set_ylabel("Count")
        ax.set_title("Loan Approval Distribution")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.dataframe(missing.rename("Missing Count").to_frame(), use_container_width=True)
        else:
            st.success("No missing values found.")

    elif page == "Model Performance":
        st.header("Model Performance")

        models = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}
        cols = st.columns(3)
        results = {}
        for idx, (name, model) in enumerate(models.items()):
            res = evaluate_model(model, X_test_s, y_test, name)
            results[name] = res
            with cols[idx]:
                st.subheader(name)
                st.metric("Accuracy", f"{res['accuracy']*100:.2f}%")

        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (name, res) in zip(axes, results.items()):
            sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Denied", "Approved"], yticklabels=["Denied", "Approved"], ax=ax)
            ax.set_title(name)
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Accuracy Comparison")
        acc_df = pd.DataFrame({"Model": list(results.keys()),
                               "Accuracy": [r["accuracy"] for r in results.values()]})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(acc_df["Model"], acc_df["Accuracy"] * 100, color=["steelblue", "salmon", "teal"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")
        for i, v in enumerate(acc_df["Accuracy"]):
            ax.text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center")
        st.pyplot(fig)
        plt.close(fig)

    elif page == "Predict Eligibility":
        st.header("Predict Loan Eligibility")
        st.markdown("Fill in the applicant details to predict loan approval.")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Income ($/month)", min_value=0, max_value=100000, value=5000)
        with col2:
            coapplicant_income = st.number_input("Co-applicant Income ($/month)", min_value=0.0, max_value=100000.0, value=0.0)
            loan_amount = st.number_input("Loan Amount ($1000)", min_value=1.0, max_value=1000.0, value=150.0)
            loan_term = st.selectbox("Loan Amount Term (months)", [360.0, 180.0, 480.0, 300.0, 84.0, 240.0, 120.0, 60.0, 36.0, 12.0])
            credit_history = st.selectbox("Credit History", ["1.0", "0.0"], format_func=lambda x: "Good (1)" if x == "1.0" else "Bad (0)")
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        model_choice = st.radio("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

        if st.button("Check Eligibility", type="primary"):
            try:
                input_dict = {
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": float(loan_term),
                    "Credit_History": float(credit_history),
                    "Gender_Female": 1 if gender == "Female" else 0,
                    "Gender_Male": 1 if gender == "Male" else 0,
                    "Married_No": 1 if married == "No" else 0,
                    "Married_Yes": 1 if married == "Yes" else 0,
                    "Dependents_0": 1 if dependents == "0" else 0,
                    "Dependents_1": 1 if dependents == "1" else 0,
                    "Dependents_2": 1 if dependents == "2" else 0,
                    "Dependents_3+": 1 if dependents == "3+" else 0,
                    "Education_Graduate": 1 if education == "Graduate" else 0,
                    "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
                    "Self_Employed_No": 1 if self_employed == "No" else 0,
                    "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
                    "Property_Area_Rural": 1 if property_area == "Rural" else 0,
                    "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
                    "Property_Area_Urban": 1 if property_area == "Urban" else 0,
                }
                input_df = pd.DataFrame([input_dict])
                # Align columns to match training
                for col in feature_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_cols]
                input_scaled = scaler.transform(input_df)
                model = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}[model_choice]
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None
                if prediction == 1:
                    st.success("✅ **Loan APPROVED** — This applicant is eligible for the loan.")
                else:
                    st.error("❌ **Loan DENIED** — This applicant does not meet the eligibility criteria.")
                if proba is not None:
                    st.write(f"Confidence: Denied={proba[0]*100:.1f}%  Approved={proba[1]*100:.1f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.error("Prediction error: %s", e)


if __name__ == "__main__":
    main()
