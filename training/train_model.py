import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("dataset/job_contract_scam_dataset.csv")
df = df.drop(columns=["posting_id"])

# Separate features and target
X = df.drop(columns=["is_fraudulent"])
y = df["is_fraudulent"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Text columns
text_columns = [
    "title",
    "company_name",
    "contract_type",
    "experience_letter_terms",
    "package_benefits_detail"
]

X_train_text = X_train[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
X_test_text = X_test[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

# Convert text into numerical features
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Numerical features
numeric_columns = [
    "contract_withholding_clause",
    "has_official_email"
]

X_train_numeric = X_train[numeric_columns].astype(float).values
X_test_numeric = X_test[numeric_columns].astype(float).values

# Combine features
X_train_final = hstack([X_train_tfidf, X_train_numeric])
X_test_final = hstack([X_test_tfidf, X_test_numeric])

# Train model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_final, y_train)

print("Model training completed successfully!")

# Evaluate model
y_pred = model.predict(X_test_final)

print("\nModel Accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraudulent"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
Path("model").mkdir(exist_ok=True)

joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel and TF-IDF vectorizer saved successfully!")
