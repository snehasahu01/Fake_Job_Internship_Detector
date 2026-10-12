
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("dataset/job_contract_scam_dataset.csv")
df = df.drop(columns=["posting_id"])

X = df.drop(columns=["is_fraudulent"])
y = df["is_fraudulent"]

# Identical job details receive the same group ID
groups = pd.util.hash_pandas_object(X, index=False).values

# Split so identical job details cannot appear in both sets
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

# Text features
text_columns = [
    "title",
    "company_name",
    "contract_type",
    "experience_letter_terms",
    "package_benefits_detail"
]

train_text = X_train[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
test_text = X_test[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)

# Numeric features
numeric_columns = [
    "contract_withholding_clause",
    "has_official_email"
]

train_numeric = X_train[numeric_columns].astype(float).values
test_numeric = X_test[numeric_columns].astype(float).values

# Combine features
X_train_final = hstack([train_tfidf, train_numeric])
X_test_final = hstack([test_tfidf, test_numeric])

# Train model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

print("\nGroup-aware validation accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Genuine", "Fraudulent"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
