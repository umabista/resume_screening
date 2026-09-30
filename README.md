# Resume Screening System

This project is a Flask-based resume screening application that compares uploaded resumes against a job description and ranks the best matches using TF-IDF vectorization and cosine similarity.

## Features
- Upload a resume in PDF format
- Enter a job description
- Extract text from the uploaded PDF
- Clean and normalize resume text
- Match resumes against the given job requirements
- Display the similarity score and shortlist recommendation
- Show matched and missing skills

## Project Structure
- `main.py` — main application logic and Flask routes
- `app.py` — minimal app entry point
- `dataset/resumes.csv` — sample resume dataset
- `templates/` — HTML templates for the frontend
- `requirements.txt` — project dependencies

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python main.py
   ```
4. Open the app in a browser at `http://127.0.0.1:5000`.

## Tech Stack
- Python
- Flask
- Pandas
- scikit-learn
- pypdf

## Notes
The app is configured to run on `0.0.0.0` and uses the `PORT` environment variable when available, which is suitable for hosted deployments.
