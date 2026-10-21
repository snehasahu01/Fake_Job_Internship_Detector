# JobShield AI — Fake Job and Internship Detector

## Project Overview
JobShield AI is a machine learning project that classifies job postings as **Genuine** or **Potentially Fraudulent**. It uses text-based features and job-related information to generate a prediction.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- FastAPI
- HTML, CSS, JavaScript

## Features
- Dataset inspection and preprocessing
- Text feature extraction using TF-IDF
- Machine learning model training and evaluation
- FastAPI prediction endpoint
- Responsive web frontend
- Genuine and fraudulent probability display
- Manual job verification tips

## Project Structure
- `main.py` — FastAPI backend
- `frontend/index.html` — Web interface
- `training/train_model.py` — Model training script
- `model/` — Trained model and TF-IDF vectorizer files
- `requirements.txt` — Python dependencies

## Run Locally

1. Install the dependencies:

   `pip install -r requirements.txt`

2. Make sure the trained model files are available in the `model/` directory.

3. Start the API:

   `python -m uvicorn main:app --reload`

4. Open the API documentation:

   `http://127.0.0.1:8000/docs`

5. Open the frontend using VS Code Live Server, usually at:

   `http://127.0.0.1:5500`

## Important Disclaimer
This project provides an ML-based risk prediction, not proof that a job posting is genuine or fraudulent. Always verify the employer, official email domain, and offer details independently.

## Project Status
The machine learning pipeline, FastAPI backend, and responsive frontend have been developed and tested locally. Deployment is pending.
