import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request
from pypdf import PdfReader

app = Flask(__name__)
# LOAD DATASET
data = pd.read_csv("dataset/resumes.csv")
print(data)
# CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text
data["clean_resume"] = data["resume"].apply(clean_text)

print(data[["resume", "clean_resume"]])
# TF-IDF


vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(data["clean_resume"])
# EXTRACT TEXT FROM PDF
def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text
# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")
# CHECK MATCH
@app.route("/check", methods=["POST"])
def check_match():

    # Get uploaded PDF
    resume_file = request.files.get("resume")

    # Get job description
    job_description = request.form.get("job_description", "")

    
    # VALIDATION
    

    if not resume_file:
        return "Please upload a resume PDF."

    if not job_description.strip():
        return "Please enter a job description."

    
    # EXTRACT RESUME TEXT
    

    try:

        resume_text = extract_pdf_text(resume_file)

    except Exception as e:

        return f"Error reading PDF: {e}"

    if not resume_text.strip():

        return "Could not extract text from this PDF."

    
    # CLEAN TEXT
    

    clean_resume = clean_text(resume_text)

    clean_job = clean_text(job_description)

    
    # JOB DESCRIPTION VECTOR
    

    job_vector = vectorizer.transform([clean_job])

    
    # DATASET SIMILARITY
    

    similarity = cosine_similarity(job_vector, tfidf_matrix)

    
    # BEST DATASET MATCH
    

    best_index = similarity[0].argmax()

    dataset_score = similarity[0][best_index] * 100

    
    # USER RESUME MATCH
    

    resume_vector = vectorizer.transform([clean_resume])

    user_similarity = cosine_similarity(resume_vector, job_vector)[0][0]

    match_score = user_similarity * 100

    
    # SKILLS
    

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

    matched_skills = []

    missing_skills = []

    for skill in skills:

        if skill in clean_job:

            if skill in clean_resume:

                matched_skills.append(skill)

            else:

                missing_skills.append(skill)

    
    # SHORTLIST STATUS
    

    if match_score >= 60:

        status = "SHORTLISTED ✅"

    else:

        status = "NOT SHORTLISTED ❌"

    
    # RESULT PAGE
    

    return render_template(
        "result.html",
        score=round(match_score, 2),
        status=status,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        category=(
            data.iloc[best_index]["job_category"]
            if "job_category" in data.columns
            else "Unknown"
        ),
        dataset_score=round(dataset_score, 2),
    )

# RUN APPLICATION


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
