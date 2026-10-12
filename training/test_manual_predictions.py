
import pandas as pd
import joblib
from scipy.sparse import hstack

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model/fake_job_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load our new test cases
df = pd.read_csv("training/manual_test_cases.csv")

# Text columns must match the training order
text_columns = [
    "title",
    "company_name",
    "contract_type",
    "experience_letter_terms",
    "package_benefits_detail"
]

# Convert text into TF-IDF features
text_data = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
text_features = vectorizer.transform(text_data)

# Numeric features must match the training order
numeric_columns = [
    "contract_withholding_clause",
    "has_official_email"
]
numeric_features = df[numeric_columns].fillna(0).astype(float).values

# Combine text and numeric features
X = hstack([text_features, numeric_features])

# Predict
predictions = model.predict(X)

# Display results
for i, prediction in enumerate(predictions):
    result = "Fraudulent" if int(prediction) == 1 else "Genuine"

    print("\n------------------------------")
    print("Job Title:", df.iloc[i]["title"])
    print("Company:", df.iloc[i]["company_name"])
    print("Model Prediction:", result)
    print("Expected Label: Review manually")
