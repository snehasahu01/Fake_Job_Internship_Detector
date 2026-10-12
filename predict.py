
import pandas as pd
import joblib
from scipy.sparse import hstack

# Load saved model and TF-IDF vectorizer
model = joblib.load("model/fake_job_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# Enter job details
title = input("Enter job title: ")
company_name = input("Enter company name: ")
contract_type = input("Enter contract type: ")
experience_letter_terms = input("Enter experience letter terms: ")
package_benefits_detail = input("Enter package and benefits details: ")

# Enter numeric features (0 or 1)
contract_withholding_clause = int(
    input("Contract withholding clause? (1=Yes, 0=No): ")
)
has_official_email = int(
    input("Official company email available? (1=Yes, 0=No): ")
)

# Combine text features in the same order used during training
text = (
    f"{title} {company_name} {contract_type} "
    f"{experience_letter_terms} {package_benefits_detail}"
)

text_tfidf = tfidf.transform([text])

# Combine text and numeric features
numeric_features = [[
    contract_withholding_clause,
    has_official_email
]]

final_features = hstack([
    text_tfidf,
    numeric_features
])

# Predict job category
prediction = model.predict(final_features)[0]

if prediction == 1:
    print("\nResult: FRAUDULENT JOB")
    print("Warning: This posting may be suspicious.")
else:
    print("\nResult: GENUINE JOB")
    print("The model classified this posting as genuine.")

# Display model confidence scores
probabilities = model.predict_proba(final_features)[0]
print(f"Model probability for Genuine: {probabilities[0] * 100:.2f}%")
print(f"Model probability for Fraudulent: {probabilities[1] * 100:.2f}%")
