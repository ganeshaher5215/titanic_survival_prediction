import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from features import engineer_features
# from src.features import engineer_features


st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🛳️ Titanic Survival Prediction")
st.write("Enter passenger details or upload a CSV to predict survival.")

model_path = Path(__file__).resolve().parents[1] / "models" / "model_rf_tuned.pkl"
fallback_path = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

model = None
if model_path.exists():
    model = joblib.load(model_path)
elif fallback_path.exists():
    model = joblib.load(fallback_path)

if model is None:
    st.warning("Model file not found. Please train a model (see README).")
else:
    st.success("Model loaded. Ready to predict!")

with st.expander("Manual Input"):
    col1, col2 = st.columns(2)
    with col1:
        Pclass = st.selectbox("Pclass", [1,2,3], index=1)
        Sex = st.selectbox("Sex", ["male","female"], index=0)
        Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
        SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
    with col2:
        Parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
        Fare = st.number_input("Fare", min_value=0.0, value=7.25, step=0.5)
        Embarked = st.selectbox("Embarked", ["S","C","Q"], index=0)
        Name = st.text_input("Name (optional)", value="Doe, Mr. John")
    Ticket = st.text_input("Ticket (optional)", value="A/5 21171")
    Cabin = st.text_input("Cabin (optional)", value="")

    if st.button("Predict"):
        df = pd.DataFrame([{
            "Pclass": Pclass, "Sex": Sex, "Age": Age, "SibSp": SibSp,
            "Parch": Parch, "Fare": Fare, "Embarked": Embarked,
            "Name": Name, "Ticket": Ticket, "Cabin": Cabin
        }])
        df_eng = engineer_features(df)
        if model is None:
            st.stop()
        # Ensure PassengerId exists (dummy column for old model)
        if "PassengerId" not in df_eng.columns:
            df_eng["PassengerId"] = 0

        pred = model.predict(df_eng)[0]
        proba = model.predict_proba(df_eng)[0][1] if hasattr(model,"predict_proba") else None
        st.write("**Prediction:**", "Survived ✅" if pred==1 else "Did Not Survive ❌")
        if proba is not None:
            st.write(f"**Probability of Survival:** {proba:.2%}")

st.divider()
st.subheader("📄 Batch CSV Prediction")
uploaded = st.file_uploader("Upload CSV with Titanic-like columns", type=["csv"])
if uploaded and model is not None:
    df = pd.read_csv(uploaded)
    df_eng = engineer_features(df)
    preds = model.predict(df_eng)
    out = df.copy()
    out["prediction"] = preds
    if hasattr(model, "predict_proba"):
        out["probability"] = model.predict_proba(df_eng)[:,1]
    st.write("Preview:")
    st.dataframe(out.head(20))
    st.download_button("Download Predictions CSV", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
