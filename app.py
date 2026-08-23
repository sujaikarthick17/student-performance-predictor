import json

import joblib
import pandas as pd
import streamlit as st

from src.config import METADATA_PATH, MODEL_PATH

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the performance category.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

with st.form("prediction_form"):
    values = {}
    for column in metadata["numeric_features"]:
        values[column] = st.number_input(
            column.replace("_", " ").title(),
            value=float(metadata["numeric_defaults"][column]),
        )
    for column in metadata["categorical_features"]:
        options = metadata["categorical_options"][column]
        values[column] = st.selectbox(column.replace("_", " ").title(), options)
    submitted = st.form_submit_button("Predict performance")

if submitted:
    student = pd.DataFrame([values], columns=metadata["feature_columns"])
    prediction = model.predict(student)[0]
    st.success(f"Predicted {metadata['target_column'].replace('_', ' ')}: **{prediction}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student)[0]
        probability_df = pd.DataFrame({"Class": model.classes_, "Probability": probabilities})
        st.bar_chart(probability_df.set_index("Class"))

st.caption(f"Test accuracy when trained: {metadata['test_accuracy']:.2%}")
