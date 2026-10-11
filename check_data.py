import pandas as pd

df = pd.read_csv("dataset/job_contract_scam_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nFraudulent Distribution:")
print(df["is_fraudulent"].value_counts())

print("\nFraudulent Percentage:")
print(df["is_fraudulent"].value_counts(normalize=True) * 100)

# Remove ID column
df = df.drop(columns=["posting_id"])

print("\nDataset after removing posting_id:")
print(df.head())
print("\nNew Shape:", df.shape)

# Separate features and target

X = df.drop(columns=["is_fraudulent"])
y = df["is_fraudulent"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

print("\nData Types:")
print(X.dtypes)

print("\nUnique Values:")
for column in X.columns:
    print(f"\n{column}:")
    print(X[column].unique())

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

# Combine text columns into one text feature
text_columns = [
    "title",
    "company_name",
    "contract_type",
    "experience_letter_terms",
    "package_benefits_detail"
]

X_train_text = X_train[text_columns].fillna("").agg(" ".join, axis=1)
X_test_text = X_test[text_columns].fillna("").agg(" ".join, axis=1)

print("\nCombined Training Text:")
print(X_train_text.head())
print("\nCombined Testing Text:")
print(X_test_text.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

# Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(X_train_text)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test_text)

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)

print("\nNumber of TF-IDF Features:")
print(len(tfidf.get_feature_names_out()))

from scipy.sparse import hstack, csr_matrix

# Select numeric features
numeric_columns = [
    "contract_withholding_clause",
    "has_official_email"
]

X_train_numeric = csr_matrix(
    X_train[numeric_columns].values
)

X_test_numeric = csr_matrix(
    X_test[numeric_columns].values
)

# Combine TF-IDF + numeric features
X_train_final = hstack([
    X_train_tfidf,
    X_train_numeric
])

X_test_final = hstack([
    X_test_tfidf,
    X_test_numeric
])

print("\nFinal Training Shape:")
print(X_train_final.shape)

print("\nFinal Testing Shape:")
print(X_test_final.shape)

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train the model
model.fit(X_train_final, y_train)

print("\nModel Training Completed Successfully!")

# Make predictions on test data
y_pred = model.predict(X_test_final)

print("\nPredictions Completed Successfully!")

print("\nFirst 20 Predictions:")
print(y_pred[:20])

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

from sklearn.metrics import classification_report

# Detailed model evaluation
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraudulent"]
))

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

import joblib

# Save trained model
joblib.dump(model, "fake_job_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel and TF-IDF vectorizer saved successfully!")

# Load saved model and vectorizer
loaded_model = joblib.load("fake_job_model.pkl")
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")

print("\nSaved Model Loaded Successfully!")
print("Saved TF-IDF Vectorizer Loaded Successfully!")

# Verify prediction using saved model
loaded_prediction = loaded_model.predict(X_test_final)

print("\nSaved Model Prediction Check:")
print(loaded_prediction[:20])