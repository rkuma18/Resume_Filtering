import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Resume Classifier", layout="centered")

st.title("AI Resume Classifier")
st.write("Paste your resume text below and let the model predict the most suitable job category.")

resume_text = st.text_area("📝 Paste Resume Text Here", height=300)

if st.button("🔍 Classify Resume"):
    if not resume_text.strip():
        st.warning("Please paste some resume content before submitting.")
    else:
        try:
            pipeline = PredictPipeline()
            prediction = pipeline.predict(resume_text)
            st.success(f"✅ Predicted Category: **{prediction}**")
        except Exception as e:
            st.error(f"🚫 An error occurred: {e}")
