from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jobshield-ai-frontend.onrender.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fake Job Internship Detector API",
    description="API to predict suspicious job postings",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Load the saved ML model and TF-IDF vectorizer
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "model" / "fake_job_model.pkl")
tfidf = joblib.load(BASE_DIR / "model" / "tfidf_vectorizer.pkl")


# Define the input format
class JobDetails(BaseModel):
    title: str
    company_name: str
    contract_type: str
    experience_letter_terms: str
    package_benefits_detail: str
    contract_withholding_clause: int = Field(ge=0, le=1)
    has_official_email: int = Field(ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "Fake Job Internship Detector API is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_job(job: JobDetails):
    # Combine text fields in the same order used during training
    text = (
        f"{job.title} {job.company_name} {job.contract_type} "
        f"{job.experience_letter_terms} {job.package_benefits_detail}"
    )

    text_features = tfidf.transform([text])

    # Numerical feature order must match training
    numeric_features = [[
        job.contract_withholding_clause,
        job.has_official_email
    ]]

    final_features = hstack([
        text_features,
        numeric_features
    ])

    prediction = int(model.predict(final_features)[0])
    probabilities = model.predict_proba(final_features)[0]

    return {
        "prediction": (
            "Fraudulent" if prediction == 1 else "Genuine"
        ),
        "genuine_probability": round(float(probabilities[0]) * 100, 2),
        "fraudulent_probability": round(float(probabilities[1]) * 100, 2),
        "disclaimer": "This is an ML prediction, not proof that a job is genuine or fraudulent."
    }
