import re

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def extract_pdf_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


data = pd.read_csv("dataset/resumes.csv")
data["clean_resume"] = data["resume"].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["clean_resume"])

skills = [
    "python",
    "flask",
    "sql",
    "git",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "pandas",
    "numpy",
    "docker",
    "aws",
    "data preprocessing",
    "html",
    "css",
    "javascript",
]

st.set_page_config(page_title="Resume Screening & Job Matching", layout="wide")
st.title("📄 Resume Screening & Job Matching")
st.caption("Find the right candidate for the right job")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume")
job_description = st.text_area(
    "Job Description",
    placeholder="Enter the job description here...",
    height=180,
)

if st.button("🔍 Check Match", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        try:
            resume_text = extract_pdf_text(uploaded_file)
        except Exception as exc:
            st.error(f"Error reading PDF: {exc}")
            st.stop()

        if not resume_text.strip():
            st.warning("Could not extract text from this PDF.")
            st.stop()

        clean_resume = clean_text(resume_text)
        clean_job = clean_text(job_description)

        job_vector = vectorizer.transform([clean_job])
        similarity = cosine_similarity(job_vector, tfidf_matrix)
        best_index = similarity[0].argmax()
        dataset_score = similarity[0][best_index] * 100

        resume_vector = vectorizer.transform([clean_resume])
        user_similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        match_score = user_similarity * 100

        matched_skills = []
        missing_skills = []
        for skill in skills:
            if skill in clean_job:
                if skill in clean_resume:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)

        status = "SHORTLISTED ✅" if match_score >= 60 else "NOT SHORTLISTED ❌"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Match Score", f"{round(match_score, 2)}%")
        with col2:
            st.metric("Status", status)

        st.write(f"**Recommended Job Category:** {data.iloc[best_index]['job_category']}")
        st.write(f"**Dataset Similarity:** {round(dataset_score, 2)}%")

        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("🟢 Matched Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.write(f"- {skill}")
            else:
                st.write("No matched skills found.")

        with right_col:
            st.subheader("🔴 Missing Skills")
            if missing_skills:
                for skill in missing_skills:
                    st.write(f"- {skill}")
            else:
                st.write("No missing skills listed.")